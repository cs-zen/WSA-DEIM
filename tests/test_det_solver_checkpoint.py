from pathlib import Path
from types import SimpleNamespace
import json

from engine.solver import det_solver as det_solver_module
from engine.solver.det_solver import DetSolver


class _Parameter:
    requires_grad = True

    def numel(self):
        return 1


class _Model:
    def parameters(self):
        return [_Parameter()]


class _DataLoader:
    def __init__(self):
        self.collate_fn = SimpleNamespace(
            stop_epoch=1,
            ema_restart_decay=0.9999,
        )
        self.sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_stage2_checkpoint_is_saved_after_failed_candidate_is_reloaded(
    monkeypatch,
    tmp_path,
):
    solver = object.__new__(DetSolver)
    solver.cfg = SimpleNamespace(
        lrsheduler=None,
        epoches=3,
        clip_max_norm=0.1,
        print_freq=10,
        checkpoint_freq=1,
    )
    solver.train = lambda: None
    solver.model = _Model()
    solver.criterion = object()
    solver.postprocessor = object()
    solver.train_dataloader = _DataLoader()
    solver.val_dataloader = object()
    solver.evaluator = object()
    solver.device = "cpu"
    solver.optimizer = object()
    solver.lr_scheduler = SimpleNamespace(step=lambda: None)
    solver.lr_warmup_scheduler = None
    solver.scaler = None
    solver.writer = None
    solver.output_dir = Path(tmp_path)
    solver.last_epoch = 1
    solver.ema = SimpleNamespace(module=object(), decay=0.9999)
    solver.training_state = "resumed"
    solver.state_dict = lambda: {
        "last_epoch": solver.last_epoch,
        "training_state": solver.training_state,
    }

    events = []

    def fake_train_one_epoch(*args, **kwargs):
        solver.training_state = "failed_stage2_candidate"
        events.append("train")
        return {"loss": 1.0}

    scores = iter((1.0, 0.5))

    def fake_evaluate(*args, **kwargs):
        score = next(scores)
        events.append(f"evaluate:{score}")
        return {"coco_eval_bbox": [score]}, None

    def fake_reload_stage1():
        solver.training_state = "reloaded_stage1"
        events.append("reload")

    saved = []

    def fake_save_on_master(state, path):
        events.append(f"save:{Path(path).name}")
        saved.append((dict(state), Path(path).name))

    solver._reload_stage1_preserving_epoch = fake_reload_stage1
    monkeypatch.setattr(det_solver_module, "stats", lambda cfg: (1, "stats"))
    monkeypatch.setattr(
        det_solver_module,
        "train_one_epoch",
        fake_train_one_epoch,
    )
    monkeypatch.setattr(det_solver_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        det_solver_module.dist_utils,
        "is_dist_available_and_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        det_solver_module.dist_utils,
        "is_main_process",
        lambda: False,
    )
    monkeypatch.setattr(
        det_solver_module.dist_utils,
        "save_on_master",
        fake_save_on_master,
    )

    solver.fit()

    assert [name for _, name in saved] == ["last.pth", "checkpoint0003.pth"]
    assert all(
        state["training_state"] == "reloaded_stage1"
        for state, _ in saved
    )
    assert events.index("reload") < events.index("save:last.pth")


def test_budget_pause_saves_epoch30_without_shortening_300_epoch_schedule(monkeypatch, tmp_path):
    solver = object.__new__(DetSolver)
    solver.cfg = SimpleNamespace(
        lrsheduler="flatcosine", epoches=300, lr_gamma=0.5,
        warmup_iter=2000, flat_epoch=144, no_aug_epoch=20,
        clip_max_norm=0.1, print_freq=10, checkpoint_freq=30,
        yaml_cfg={"stop_after_epoch": 30},
    )
    solver.train = lambda: None
    solver.model = _Model()
    solver.criterion = solver.postprocessor = solver.evaluator = object()

    class Loader(_DataLoader):
        def __len__(self):
            return 10

    solver.train_dataloader = Loader()
    solver.train_dataloader.collate_fn.stop_epoch = 280
    solver.val_dataloader = object()
    solver.device = "cpu"
    solver.optimizer = object()
    solver.lr_warmup_scheduler = solver.scaler = solver.writer = None
    solver.output_dir = Path(tmp_path)
    solver.last_epoch = -1
    solver.ema = SimpleNamespace(module=object(), decay=0.9999)
    solver.state_dict = lambda: {"last_epoch": solver.last_epoch}
    epochs, saved, scheduler_settings = [], [], {}

    def scheduler(*args, **kwargs):
        scheduler_settings.update(kwargs)
        return object()

    def train_epoch(*args, **kwargs):
        epochs.append(args[7])
        return {"loss": 1.0}

    def evaluate(*args, **kwargs):
        return {"coco_eval_bbox": [float(solver.last_epoch + 1)]}, None

    monkeypatch.setattr(det_solver_module, "FlatCosineLRScheduler", scheduler)
    monkeypatch.setattr(det_solver_module, "stats", lambda cfg: (1, "stats"))
    monkeypatch.setattr(det_solver_module, "train_one_epoch", train_epoch)
    monkeypatch.setattr(det_solver_module, "evaluate", evaluate)
    monkeypatch.setattr(det_solver_module.dist_utils, "is_dist_available_and_initialized", lambda: False)
    monkeypatch.setattr(det_solver_module.dist_utils, "is_main_process", lambda: True)
    monkeypatch.setattr(det_solver_module.dist_utils, "save_on_master",
                        lambda state, path: saved.append((dict(state), Path(path).name)))

    solver.fit()

    assert epochs == list(range(30))
    assert scheduler_settings["total_epochs"] == solver.cfg.epoches == 300
    assert scheduler_settings["warmup_iter"] == 2000
    assert scheduler_settings["flat_epochs"] == 144
    assert saved[-2:] == [({"last_epoch": 29}, "last.pth"),
                          ({"last_epoch": 29}, "checkpoint0030.pth")]
    rows = [json.loads(line) for line in (tmp_path / "log.txt").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == list(range(30))
    pause = json.loads((tmp_path / "planned_pause.json").read_text())
    assert pause["completed_epochs"] == 30
    assert pause["configured_total_epochs"] == 300
