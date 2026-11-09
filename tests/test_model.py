import copy

import torch
from scipy.optimize import linear_sum_assignment

from engine.core import YAMLConfig
from engine.deim.fsff import FSFF
from engine.backbone.stem_pdc import PDCResidualStem
from engine.deim.irpe import IRPEMultiheadAttention
from engine.optim import ModelEMA


def test_full_model_four_scales_and_deployed_inference():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = YAMLConfig('configs/wsa_deim_visdrone.yml')
    model = cfg.model.eval()
    assert isinstance(model.backbone.stem, PDCResidualStem)
    assert isinstance(model.encoder.encoder[0].layers[0].self_attn, IRPEMultiheadAttention)
    assert model.encoder.f5_branch_channels == (64, 128, 64)
    assert sum(isinstance(m, FSFF) for m in model.modules()) == 1
    shapes = []
    handle = model.decoder.register_forward_pre_hook(
        lambda module, args: shapes.extend(tuple(x.shape) for x in args[0]))
    image = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(image)
        handle.remove()
        deployed = copy.deepcopy(model).deploy()(image)
    assert shapes == [(1, 256, s, s) for s in (160, 80, 40, 20)]
    assert outputs['pred_logits'].shape == (1, 300, 10)
    assert outputs['pred_boxes'].shape == (1, 300, 4)
    for key in outputs:
        assert torch.isfinite(outputs[key]).all()
        assert torch.isfinite(deployed[key]).all()
    cost = (outputs['pred_boxes'][0, :, None] - deployed['pred_boxes'][0, None]).abs().sum(-1)
    row, col = linear_sum_assignment(cost.numpy())
    for key in outputs:
        torch.testing.assert_close(outputs[key][0, row], deployed[key][0, col], rtol=2e-4, atol=2e-5)


def test_synthetic_complete_loss_backward_optimizer_and_ema():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = YAMLConfig('configs/wsa_deim_visdrone.yml')
    model = cfg.model.train()
    images = torch.rand(2, 3, 128, 128)
    targets = [{'labels': torch.tensor([1, 4]),
                'boxes': torch.tensor([[0.3, 0.4, 0.2, 0.15], [0.7, 0.6, 0.1, 0.2]])}
               for _ in range(2)]
    optimizer = cfg.optimizer
    ema = ModelEMA(model, decay=0.9999, warmups=1000)
    before = model.encoder.fpn.fsff.attention.output_projection.conv.weight.detach().clone()
    losses = cfg.criterion(model(images, targets), targets, epoch=0, step=0, global_step=0,
                           epoch_step=1)
    assert losses and all(torch.isfinite(v).all() for v in losses.values())
    total = sum(losses.values())
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
    optimizer.step()
    ema.update(model)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.equal(before, model.encoder.fpn.fsff.attention.output_projection.conv.weight)
    assert ema.updates == 1
