# VisDrone data preparation

## Source and fixed revision

This recipe uses the Voxel51 FiftyOne export of VisDrone2019-DET, pinned to
revision `3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2`:

- [Pinned dataset snapshot](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET/tree/3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2)
- [Pinned dataset card and terms](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET/blob/3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2/README.md)
- [Original VisDrone project](https://github.com/VisDrone/VisDrone-Dataset)

Download `samples.json` and the `data/` image directory from that revision to a
local source directory. One option, using Git LFS, is:

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/Voxel51/VisDrone2019-DET /path/to/VisDrone-source
git -C /path/to/VisDrone-source checkout 3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2
git -C /path/to/VisDrone-source lfs pull --include="samples.json,data/*"
git -C /path/to/VisDrone-source lfs fsck
```

Replace `/path/to/VisDrone-source` with a new local directory. Obtain the actual
LFS content, not pointer files. No FiftyOne installation is needed for the
conversion utility. The code license does not license the dataset; consult
the original project and pinned dataset card for data-use terms.

## Generate the training layout

From the WSA-DEIM repository root:

```bash
python tools/prepare_visdrone.py \
  --samples-json /path/to/VisDrone-source/samples.json \
  --images-dir /path/to/VisDrone-source/data \
  --output-dir data/VisDrone2019-DET
```

The default copies images for the `train` and `val` splits. It checks image
dimensions against the source metadata and image readability before creating
the output directory. For byte-level image integrity, use the source snapshot's
Git LFS checks. Use `--image-mode symlink` to avoid copying image bytes; keep the
source directory at its original location while using those links.

```text
data/VisDrone2019-DET/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── images/
│   ├── train/
│   └── val/
└── preparation.json
```

To include test-dev, add `--splits train val test-dev` on the first preparation
call. To prepare only validation, use `--splits val`. Each invocation requires
a new output directory and refuses to overwrite an existing one. A partial
directory left by an I/O failure is retained for diagnosis.

To verify the labels without downloading or organizing images:

```bash
python tools/prepare_visdrone.py \
  --samples-json /path/to/VisDrone-source/samples.json \
  --annotations-only --splits train val test-dev \
  --output-dir /path/to/new-annotation-check
```

Annotation-only output is not a training-ready dataset. The report explicitly
records that images were not checked.

## Exact conversion policy

The utility preserves the recorded export rather than introducing a new
conversion from raw VisDrone TXT files:

1. Source tags `train`, `val`, and `test` become `train`, `val`, and `test-dev`.
   No samples are resplit or moved between training and validation.
2. Samples in each split are sorted by their source `filepath`. Image IDs start
   at 1 in each split. Annotation IDs continue across train, val, then test-dev,
   even when only one split is requested for output.
3. Normalized `[x, y, width, height]` is multiplied by the recorded image
   dimensions. Coordinates are not rounded or clipped during export.
4. The source labels `pedestrians` and `awning_tricycle` are mapped to
   `pedestrian` and `awning-tricycle`. Labels outside the ten target categories
   are excluded; in this pinned snapshot they are `ignore_regions` and `others`.
5. Non-positive box width or height is excluded. Empty images are retained.
   Truncation and occlusion attributes are retained without additional filtering.
6. Retained annotations have `iscrowd=0`, `segmentation=[]`, and `area=width*height`.

The training loader performs its existing clipping and augmentation afterward.
Discarding ignored-region annotations does **not** implement official
ignored-region evaluation: detections in those regions are not specially
suppressed or ignored by this COCO evaluator.

| COCO ID | Training/inference label | Category |
| --- | --- | --- |
| 1 | 0 | pedestrian |
| 2 | 1 | people |
| 3 | 2 | bicycle |
| 4 | 3 | car |
| 5 | 4 | van |
| 6 | 5 | truck |
| 7 | 6 | tricycle |
| 8 | 7 | awning-tricycle |
| 9 | 8 | bus |
| 10 | 9 | motor |

## Expected export

| Split | Images | Retained boxes | Ignored-region boxes removed | Other boxes removed | Degenerate boxes removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 6471 | 343204 | 8813 | 1532 | 1 |
| val | 548 | 38759 | 1378 | 32 | 0 |
| test-dev | 1610 | 75102 | 2180 | 265 | 0 |

The source and annotation SHA256 values are stored in
[visdrone_export_manifest.json](../configs/visdrone_export_manifest.json).
All three annotation hashes are verified before any output is written. A
different source file or conversion result is rejected rather than silently
accepted. The generated `preparation.json` records counts and hashes for each
requested split. Matching these labels does not establish model accuracy or
checkpoint compatibility.
