# WSA-DEIM

Official repository for WSA-DEIM.

## Install

Use Python 3.11. Run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The model dependencies pin PyTorch 2.8.0 and torchvision 0.23.0. CUDA training
requires compatible NVIDIA hardware, drivers and a CUDA-enabled PyTorch build.
Image inference also supports CPU. Video inference additionally requires:

```bash
python -m pip install -r requirements-video.txt
```

## Prepare VisDrone

Follow [the data preparation guide](docs/DATA.md) to obtain the pinned dataset
snapshot. The converter verifies the source SHA256 and reproduces the recorded
COCO annotation files byte for byte. It does not download or redistribute data.

```bash
python tools/prepare_visdrone.py \
  --samples-json /path/to/VisDrone-source/samples.json \
  --images-dir /path/to/VisDrone-source/data \
  --output-dir data/VisDrone2019-DET
```

The output directory must be new. The default prepares `train` and `val`, copies
their images, and records the annotation hashes in `preparation.json`. Use
`--image-mode symlink` to link images while keeping the source snapshot available.
The optional `test-dev` split and annotation-only verification are documented
in the data guide.

The generated layout matches [the configuration](configs/wsa_deim_visdrone.yml).
For another location, edit the two dataloaders' `img_folder` and `ann_file` paths.
Category IDs 1–10 are mapped to training labels 0–9 and mapped back for evaluation.

## Train and evaluate

The supplied configuration uses random initialization, seed 0, 640 × 640 inputs,
one GPU, training batch size 4, validation batch size 8, 300 epochs, AMP, EMA,
and gradient clipping at 0.1. AdamW uses 1e-4 for the backbone and 2e-4 elsewhere.

Training:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/wsa_deim_visdrone.yml --use-amp --seed 0
```

Resume:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/wsa_deim_visdrone.yml --use-amp --seed 0 \
  -r outputs/wsa_deim_visdrone_seed0/last.pth
```

Evaluation:

```bash
python train.py -c configs/wsa_deim_visdrone.yml --test-only --seed 0 \
  -r outputs/wsa_deim_visdrone_seed0/last.pth \
  --output-dir outputs/evaluation
```

## Inference

Supply a checkpoint trained with this implementation. Loading is strict; EMA
weights are selected when present.

Image inference:

```bash
python infer.py -r /path/to/checkpoint.pth -i /path/to/image.jpg \
  -o outputs/prediction.jpg --json-output outputs/prediction.json --device cuda:0
```

Video inference:

```bash
python infer.py -r /path/to/checkpoint.pth -i /path/to/video.mp4 \
  -o outputs/prediction.mp4 --device cuda:0
```

## License and acknowledgements

Source code is released under [Apache-2.0](LICENSE). Original source notices and
modification notices are retained; see [third-party notices](THIRD_PARTY_NOTICES.md).
Dataset terms are separate from the code license.

This implementation builds on [DEIM](https://github.com/Intellindust-AI-Lab/DEIM),
[D-FINE](https://github.com/Peterande/D-FINE), and
[RT-DETR](https://github.com/lyuwenyu/RT-DETR). The contextual iRPE-K code is adapted
from [Microsoft Cream](https://github.com/microsoft/Cream/tree/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/iRPE/DETR-with-iRPE).
VisDrone was created by the AISKYEYE team; the exact data export is provided by
[Voxel51](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET).
