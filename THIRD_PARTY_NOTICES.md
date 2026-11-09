# Third-party notices

## DEIM, D-FINE and RT-DETR

WSA-DEIM is derived from DEIM (upstream revision
`09d35d53d39ee3145a1e61e3a989b28b9468d1dd`) and retains its Apache-2.0 license.
Source headers identify D-FINE, RT-DETR, PaddleDetection and torchvision origins.
See `LICENSE` and the copyright notices in the source files.

## Microsoft Cream iRPE

The contextual Product, shared-head, K-only image relative position encoding
used by the F5 AIFI attention path is adapted from the official DETR iRPE
implementation in Microsoft Cream:
<https://github.com/microsoft/Cream/tree/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/iRPE/DETR-with-iRPE>.

Pinned upstream revision:
`4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9`

Principal reference sources:

- <https://github.com/microsoft/Cream/blob/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/iRPE/DETR-with-iRPE/models/transformer.py>
- <https://github.com/microsoft/Cream/blob/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/iRPE/DETR-with-iRPE/models/rpe_attention/irpe.py>
- <https://github.com/microsoft/Cream/blob/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/iRPE/DETR-with-iRPE/models/rpe_attention/rpe_attention_function.py>

The upstream DETR-with-iRPE code is licensed under the Apache License,
Version 2.0.  Its license text is available at:
<https://github.com/microsoft/Cream/blob/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/iRPE/DETR-with-iRPE/LICENSE>.

## VisDrone data provenance

VisDrone was created by the AISKYEYE team at Tianjin University. The data
preparation utility targets the Voxel51 FiftyOne export of VisDrone2019-DET at
revision `3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2`.

- Original project: <https://github.com/VisDrone/VisDrone-Dataset>
- Pinned export and dataset terms:
  <https://huggingface.co/datasets/Voxel51/VisDrone2019-DET/blob/3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2/README.md>

This source package does not redistribute images or annotations. The Apache-2.0
code license does not replace the dataset's separate terms. Conversion policy,
category mapping and recorded hashes are documented in `docs/DATA.md`.
