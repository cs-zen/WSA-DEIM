import torch
from torch import nn
from engine.deim.fsff import FSFF, GLCAM

def test_glcam_shared_projection_consumes_local_plus_global():
    torch.set_num_threads(2)
    attention = GLCAM(32).double().eval()
    x = torch.randn(2, 32, 11, 13, dtype=torch.float64, requires_grad=True)
    local = attention.local_branch(x)
    z = attention.global_projection(x)
    freq, multiscale, identity = z.split((4, 4, 8), dim=1)
    spectrum = torch.fft.fft2(freq, norm='ortho')
    filtered = attention.frequency_depthwise(torch.cat((spectrum.real, spectrum.imag), dim=1))
    real, imag = filtered.chunk(2, dim=1)
    vf = torch.fft.ifft2(torch.complex(real, imag), norm='ortho').real
    shared = attention.multi_scale_pre(multiscale)
    vm = shared + sum(branch(shared) for branch in attention.multi_scale_branches)
    expected_input = local + torch.cat((vf, vm, identity), dim=1) + z
    captured = []
    handle = attention.output_projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    actual = attention(x)
    handle.remove()
    assert len(captured) == 1 and captured[0].shape[1] == 16
    torch.testing.assert_close(captured[0], expected_input)
    torch.testing.assert_close(actual, torch.sigmoid(attention.output_projection(expected_input)))
    assert attention.local_branch[-1].conv.out_channels == 16
    assert attention.output_projection.conv.out_channels == 32
    assert (actual > 0).all() and (actual < 1).all()
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_fsff_owns_sni_and_complementary_fusion():
    class Mask(nn.Module):
        def forward(self, x):
            self.input = x
            return torch.full_like(x, 0.25)

    fusion = FSFF(32)
    fusion.attention = Mask()
    shallow, deep = torch.randn(2, 32, 10, 14), torch.randn(2, 32, 5, 7)
    deep_up = deep.repeat_interleave(2, -2).repeat_interleave(2, -1) / 4
    actual = fusion(shallow, deep)
    torch.testing.assert_close(fusion.attention.input, shallow + deep_up)
    torch.testing.assert_close(actual, 0.25 * shallow + 0.75 * deep_up)
