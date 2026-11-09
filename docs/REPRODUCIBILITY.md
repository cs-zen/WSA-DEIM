# Model and reproducibility notes

## Architecture

This package contains the complete PDC + DAII + SACFF architecture using
HGNetv2-B0, four feature scales P2–P5, and a three-layer decoder. DAII allocates
64 / 128 / 64 channels to identity / iRPE-K attention / HFEM, respectively.
The current complete SACFF uses convolution fusion and post-fusion MSCA.
Its separate RepNCSPELAN4 alignment-ablation architecture is not included.

The P3-to-P2 FSFF computes `D = nearest_upsample(deep, 2) / 4`, `U = shallow + D`,
and `output = M * shallow + (1 - M) * D`. In GL-CAM, the global branches are
concatenated and added to the C/2-channel projected input `Z_g`. Local and global
responses are added before a shared C/2-to-C output projection and sigmoid.

Checkpoints from `P2AFF` / `FrequencyMultiScaleAttention` implementations with
separate branch output projections are incompatible with this FSFF version.
Inference and full-checkpoint resume/evaluation use strict model-state loading.

## Validation

The following checks passed on CPU with Python 3.11.15, PyTorch 2.8.0 and
torchvision 0.23.0:

| Check | Result |
| --- | --- |
| Software suite | 31 tests passed |
| Train / val / test-dev annotation conversion from the pinned source | All three outputs matched the recorded SHA256 byte for byte |
| Real validation image preparation | 548 images passed dimension/readability checks; generated links were usable |

The tests cover inference, synthetic-data training, category mapping, checkpoint
handling and conversion. Real-data CUDA AMP validation and trained detection
accuracy have not been verified for this source version. No trained checkpoint
or performance result is associated with it.

## Source integrity and provenance

The release archive contains `SHA256SUMS` for all other packaged files. After
extracting it, verify from the repository root with `sha256sum -c SHA256SUMS`
(Linux) or `shasum -a 256 -c SHA256SUMS` (macOS).

DEIM upstream revision: `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`.
The iRPE implementation is adapted from Microsoft Cream revision
`4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9`.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the original source
headers for attribution. Dataset provenance is recorded separately in
[DATA.md](DATA.md).
