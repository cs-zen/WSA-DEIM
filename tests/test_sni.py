import pytest
import torch
import torch.nn.functional as F

from engine.core import GLOBAL_CONFIG
from engine.deim.sni import SNI


def test_sni_matches_scaled_nearest_interpolation_and_preserves_spatial_sum():
    feature = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    upsampler = SNI(up_factor=2)

    output = upsampler(feature)
    expected = F.interpolate(feature, scale_factor=2, mode='nearest') * 0.25

    assert output.shape == torch.Size([2, 3, 8, 10])
    assert upsampler.up_factor == 2
    assert upsampler.alpha == 0.25
    assert sum(parameter.numel() for parameter in upsampler.parameters()) == 0
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        output.sum(dim=(-2, -1)),
        feature.sum(dim=(-2, -1)),
    )


def test_sni_backpropagates_through_nearest_replication_and_scaling():
    feature = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)

    SNI(up_factor=2)(feature).sum().backward()

    torch.testing.assert_close(feature.grad, torch.ones_like(feature))


def test_sni_supports_other_positive_integer_factors():
    feature = torch.randn(1, 2, 2, 3)
    upsampler = SNI(up_factor=3)

    output = upsampler(feature)

    assert output.shape == torch.Size([1, 2, 6, 9])
    assert upsampler.alpha == pytest.approx(1.0 / 9.0)


@pytest.mark.parametrize('up_factor', [0, -1, 1.5, True])
def test_sni_rejects_invalid_up_factor(up_factor):
    with pytest.raises(ValueError, match='positive integer'):
        SNI(up_factor=up_factor)


def test_sni_is_registered():
    assert GLOBAL_CONFIG['SNI']['_name'] == 'SNI'
