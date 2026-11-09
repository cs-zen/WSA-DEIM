from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils, stats
from engine.optim.lr_scheduler import FlatCosineLRScheduler
from engine.solver import TASKS


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _iter_tensors(value: Any, path: str = "output"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{path}[{index}]")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise FloatingPointError(f"non-finite tensor: {name}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _optimizer_groups(optimizer, model) -> list[dict[str, Any]]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        names = [parameter_names[id(parameter)] for parameter in group["params"]]
        groups.append(
            {
                "index": index,
                "lr": float(group["lr"]),
                "initial_lr": float(group["initial_lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_tensors": len(names),
                "parameter_elements": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
                "first_parameter": names[0] if names else None,
                "last_parameter": names[-1] if names else None,
            }
        )
    return groups


def _gradient_summary(model) -> dict[str, Any]:
    tensors = 0
    elements = 0
    finite_abs_max = 0.0
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        tensors += 1
        elements += gradient.numel()
        finite_mask = torch.isfinite(gradient)
        if bool(finite_mask.all()):
            finite_abs_max = max(finite_abs_max, float(gradient.abs().max()))
            continue
        finite_values = gradient[finite_mask]
        if finite_values.numel():
            finite_abs_max = max(
                finite_abs_max,
                float(finite_values.abs().max()),
            )
        nonfinite.append(
            {
                "name": name,
                "nan": int(torch.isnan(gradient).sum()),
                "posinf": int(torch.isposinf(gradient).sum()),
                "neginf": int(torch.isneginf(gradient).sum()),
            }
        )
    return {
        "finite": not nonfinite,
        "tensors": tensors,
        "elements": elements,
        "finite_abs_max": finite_abs_max,
        "nonfinite": nonfinite,
    }


def run_one_batch(args: argparse.Namespace, evidence_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AMP smoke test")
    if dist_utils.get_world_size() != 1:
        raise RuntimeError(
            f"smoke test requires world_size=1, got {dist_utils.get_world_size()}"
        )

    config_path = Path(args.config).resolve()
    config_overrides = {
        "seed": args.seed,
        "use_amp": True,
        "output_dir": str(evidence_dir),
    }
    if args.scaler_init_scale is not None:
        config_overrides["scaler"] = {"init_scale": args.scaler_init_scale}
    cfg = YAMLConfig(str(config_path), **config_overrides)
    if cfg.tuning or cfg.resume:
        raise RuntimeError("scratch smoke test must not use tuning or resume weights")
    if cfg.seed != args.seed:
        raise RuntimeError(f"seed mismatch: {cfg.seed} != {args.seed}")
    if not cfg.use_amp or not cfg.use_ema:
        raise RuntimeError("the formal configuration must enable both AMP and EMA")

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    profiled_parameters, model_stats = stats(cfg)
    if solver.scaler is None or not solver.scaler.is_enabled():
        raise RuntimeError("CUDA GradScaler is not enabled")
    if solver.ema is None:
        raise RuntimeError("EMA was not constructed")

    loader = solver.train_dataloader
    loader.set_epoch(0)
    if dist_utils.is_dist_available_and_initialized():
        loader.sampler.set_epoch(0)

    scheduler = FlatCosineLRScheduler(
        solver.optimizer,
        cfg.lr_gamma,
        len(loader),
        total_epochs=cfg.epoches,
        warmup_iter=cfg.warmup_iter,
        flat_epochs=cfg.flat_epoch,
        no_aug_epochs=cfg.no_aug_epoch,
    )

    device = solver.device
    model = solver.model
    criterion = solver.criterion
    optimizer = solver.optimizer
    scaler = solver.scaler
    ema = solver.ema
    model.train()
    criterion.train()
    optimizer.zero_grad()
    trainable_tensors = 0
    trainable_elements = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        trainable_elements += parameter.numel()

    ema_updates_before = int(ema.updates)
    torch.cuda.reset_peak_memory_stats(device)
    loader_iterator = iter(loader)
    maximum_batches = args.max_smoke_batches if args.allow_amp_backoff else 1
    attempts = []
    successful = None

    for step in range(maximum_batches):
        try:
            samples, targets = next(loader_iterator)
        except StopIteration as error:
            raise RuntimeError("training loader ended before smoke passed") from error

        expected_batch_size = cfg.yaml_cfg["train_dataloader"]["total_batch_size"]
        if len(targets) != expected_batch_size:
            raise RuntimeError(
                "real batch size does not match train_dataloader.total_batch_size: "
                f"{len(targets)} != {expected_batch_size}"
            )
        image_ids = []
        for index, target in enumerate(targets):
            labels = target.get("labels")
            boxes = target.get("boxes")
            if labels is None or boxes is None:
                raise ValueError(f"target {index} is missing labels or boxes")
            if labels.numel() and (
                int(labels.min()) < 0
                or int(labels.max()) >= cfg.yaml_cfg["num_classes"]
            ):
                raise ValueError(f"target {index} contains an out-of-range class label")
            _require_finite(f"target[{index}].boxes", boxes)
            if "image_id" in target and target["image_id"].numel():
                image_ids.append(int(target["image_id"].reshape(-1)[0]))

        samples = samples.to(device)
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        optimizer_groups_before = _optimizer_groups(optimizer, model)
        scale_before = float(scaler.get_scale())

        with torch.autocast(device_type=str(device), cache_enabled=True):
            outputs = model(samples, targets=targets)
        for name, tensor in _iter_tensors(outputs):
            _require_finite(name, tensor)

        metas = {
            "epoch": 0,
            "step": step,
            "global_step": step,
            "epoch_step": len(loader),
        }
        with torch.autocast(device_type=str(device), enabled=False):
            loss_dict = criterion(outputs, targets, **metas)
        if not loss_dict:
            raise RuntimeError("criterion returned no component losses")
        for name, component in loss_dict.items():
            _require_finite(f"loss.{name}", component)
        total_loss = sum(loss_dict.values())
        _require_finite("loss.total", total_loss)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradients = _gradient_summary(model)
        if gradients["tensors"] == 0:
            raise RuntimeError("backward pass produced no gradients")

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.clip_max_norm,
        )
        gradient_norm_value = float(gradient_norm)
        clipped_gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)
        scheduler.step(step, optimizer)
        torch.cuda.synchronize(device)

        changed = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            _require_finite(f"parameter.{name}", parameter)
            if not torch.equal(before[name], parameter.detach().cpu()):
                changed.append(name)
        for name, parameter in ema.module.named_parameters():
            _require_finite(f"ema_parameter.{name}", parameter)

        scale_after = float(scaler.get_scale())
        attempt = {
            "step": step,
            "image_ids": image_ids,
            "batch_size": len(targets),
            "sample_shape": list(samples.shape),
            "target_box_counts": [
                int(target["boxes"].shape[0]) for target in targets
            ],
            "total_loss": float(total_loss.detach()),
            "component_losses": {
                name: float(value.detach()) for name, value in loss_dict.items()
            },
            "scaler_scale_before": scale_before,
            "scaler_scale_after": scale_after,
            "gradient_summary": gradients,
            "gradient_norm_before_clip": (
                gradient_norm_value
                if math.isfinite(gradient_norm_value)
                else str(gradient_norm_value)
            ),
            "clipped_gradients_finite": clipped_gradients_finite,
            "updated_parameter_tensors": len(changed),
            "optimizer_groups_before_step": optimizer_groups_before,
            "optimizer_groups_after_scheduler": _optimizer_groups(
                optimizer,
                model,
            ),
            "ema_updates_after": int(ema.updates),
        }

        if not gradients["finite"]:
            attempt["outcome"] = "amp_backoff"
            if changed:
                raise RuntimeError(
                    "parameters changed despite non-finite gradients and AMP backoff"
                )
            if not scale_after < scale_before:
                raise RuntimeError(
                    "GradScaler did not reduce its scale after non-finite gradients"
                )
            attempts.append(attempt)
            _write_json(evidence_dir / "attempts.json", {"attempts": attempts})
            if not args.allow_amp_backoff:
                first = gradients["nonfinite"][0]["name"]
                raise FloatingPointError(f"non-finite tensor: gradient.{first}")
            continue

        if not math.isfinite(gradient_norm_value):
            raise FloatingPointError("non-finite total gradient norm")
        if not clipped_gradients_finite:
            raise FloatingPointError("non-finite gradient after clipping")

        if not changed:
            attempt["outcome"] = "finite_no_parameter_update"
            attempts.append(attempt)
            _write_json(evidence_dir / "attempts.json", {"attempts": attempts})
            if not args.allow_amp_backoff:
                raise RuntimeError(
                    "optimizer step did not update any trainable parameter"
                )
            continue

        attempt["outcome"] = "passed"
        attempt["first_updated_parameters"] = changed[:20]
        attempts.append(attempt)
        _write_json(evidence_dir / "attempts.json", {"attempts": attempts})
        successful = {
            "samples": samples,
            "targets": targets,
            "loss_dict": loss_dict,
            "total_loss": total_loss,
            "gradients": gradients,
            "gradient_norm": gradient_norm_value,
            "changed": changed,
            "optimizer_groups_before": optimizer_groups_before,
            "scale_before": scale_before,
            "scale_after": scale_after,
        }
        break

    if successful is None:
        raise RuntimeError(
            f"no finite parameter-updating batch within {maximum_batches} attempts"
        )
    expected_ema_updates = ema_updates_before + len(attempts)
    if int(ema.updates) != expected_ema_updates:
        raise RuntimeError(
            f"EMA update count mismatch: {ema.updates} != {expected_ema_updates}"
        )

    nonfinite_ema_buffers = []
    for name, buffer in ema.module.named_buffers():
        if buffer.dtype.is_floating_point and not bool(torch.isfinite(buffer).all()):
            nonfinite_ema_buffers.append(name)

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    return {
        "status": "passed",
        "config": str(config_path),
        "config_file_sha256": _sha256_file(config_path),
        "resolved_config_sha256": _canonical_config_hash(cfg.yaml_cfg),
        "seed": args.seed,
        "world_size": dist_utils.get_world_size(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "amp_enabled": True,
        "allow_amp_backoff": args.allow_amp_backoff,
        "maximum_smoke_batches": maximum_batches,
        "attempt_count": len(attempts),
        "backoff_attempts": sum(
            attempt["outcome"] == "amp_backoff" for attempt in attempts
        ),
        "attempts": attempts,
        "scaler_scale_before_success": successful["scale_before"],
        "scaler_scale_after_success": successful["scale_after"],
        "diagnostic_scaler_init_scale_override": args.scaler_init_scale,
        "ema_updates_before": ema_updates_before,
        "ema_updates_after": int(ema.updates),
        "nonfinite_ema_buffers": nonfinite_ema_buffers,
        "clip_max_norm": float(cfg.clip_max_norm),
        "gradient_norm_before_clip": successful["gradient_norm"],
        "gradient_tensors": successful["gradients"]["tensors"],
        "gradient_elements": successful["gradients"]["elements"],
        "gradient_abs_max_before_clip": successful["gradients"][
            "finite_abs_max"
        ],
        "trainable_parameter_tensors": trainable_tensors,
        "trainable_parameter_elements": trainable_elements,
        "profiled_parameter_elements": int(profiled_parameters),
        "model_stats": str(model_stats),
        "updated_parameter_tensors": len(successful["changed"]),
        "first_updated_parameters": successful["changed"][:20],
        "batch_size": len(successful["targets"]),
        "sample_shape": list(successful["samples"].shape),
        "target_box_counts": [
            int(target["boxes"].shape[0]) for target in successful["targets"]
        ],
        "total_loss": float(successful["total_loss"].detach()),
        "component_losses": {
            name: float(value.detach())
            for name, value in successful["loss_dict"].items()
        },
        "optimizer_groups_before_step": successful["optimizer_groups_before"],
        "optimizer_groups_after_scheduler": _optimizer_groups(optimizer, model),
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scaler-init-scale", type=float)
    parser.add_argument("--allow-amp-backoff", action="store_true")
    parser.add_argument("--max-smoke-batches", type=int, default=16)
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--print-method", default="builtin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_dir = Path(args.output_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    distributed_ready = False
    try:
        distributed_ready = dist_utils.setup_distributed(
            args.print_rank,
            args.print_method,
            seed=args.seed,
        )
        if not distributed_ready:
            raise RuntimeError(
                "distributed initialization failed; launch this smoke test with "
                "torchrun --standalone --nproc_per_node=1"
            )
        report = run_one_batch(args, evidence_dir)
        if dist_utils.is_main_process():
            _write_json(evidence_dir / "report.json", report)
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        failure = {
            "status": "failed",
            "config": str(Path(args.config).resolve()),
            "seed": args.seed,
            "diagnostic_scaler_init_scale_override": args.scaler_init_scale,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if torch.cuda.is_available():
            failure["peak_gpu_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )
            failure["peak_gpu_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved()
            )
        _write_json(evidence_dir / "failure.json", failure)
        raise
    finally:
        if distributed_ready or dist_utils.is_dist_available_and_initialized():
            dist_utils.cleanup()


if __name__ == "__main__":
    main()
