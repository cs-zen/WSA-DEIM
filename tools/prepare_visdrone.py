"""Reproduce the pinned VisDrone COCO export used by this project.

The source hash and all three annotation hashes are checked before any output
is written. This utility preserves the recorded conversion, including IDs and
floating-point coordinates; it does not implement official ignored-region AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


MANIFEST = Path(__file__).resolve().parents[1] / 'configs/visdrone_export_manifest.json'
SPLITS = (('train', 'train'), ('val', 'val'), ('test-dev', 'test'))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def convert_samples(samples: list, classes: list[str]) -> dict:
    category_ids = {name: index + 1 for index, name in enumerate(classes)}
    category_ids.update(
        pedestrians=category_ids['pedestrian'],
        awning_tricycle=category_ids['awning-tricycle'],
    )
    categories = [
        {'id': index + 1, 'name': name, 'supercategory': 'object'}
        for index, name in enumerate(classes)
    ]
    annotation_id = 1
    exports = {}
    for split, source_tag in SPLITS:
        selected = sorted(
            (sample for sample in samples if source_tag in sample['tags']),
            key=lambda sample: sample['filepath'],
        )
        images, annotations = [], []
        dropped = Counter()
        for image_id, sample in enumerate(selected, 1):
            width = sample['metadata']['width']
            height = sample['metadata']['height']
            images.append({
                'id': image_id, 'file_name': Path(sample['filepath']).name,
                'width': width, 'height': height,
            })
            for detection in sample['ground_truth']['detections']:
                label = detection['label']
                if label not in category_ids:
                    dropped[label] += 1
                    continue
                x, y, box_width, box_height = detection['bounding_box']
                x, y = x * width, y * height
                box_width, box_height = box_width * width, box_height * height
                if box_width <= 0 or box_height <= 0:
                    dropped['degenerate_boxes'] += 1
                    continue
                annotations.append({
                    'id': annotation_id, 'image_id': image_id,
                    'category_id': category_ids[label],
                    'bbox': [x, y, box_width, box_height],
                    'area': box_width * box_height, 'iscrowd': 0,
                    'segmentation': [], 'truncation': detection['truncation'],
                    'occlusion': detection['occlusion'],
                })
                annotation_id += 1
        dataset = {
            'info': {'description': 'VisDrone2019-DET converted from Voxel51 FiftyOne export'},
            'licenses': [], 'images': images, 'annotations': annotations,
            'categories': categories,
        }
        encoded = json.dumps(dataset, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        exports[split] = {
            'data': encoded, 'images': images, 'annotations': len(annotations),
            'dropped': dict(dropped),
        }
    return exports


def verified_exports(samples_json: Path) -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    raw = samples_json.read_bytes()
    if sha256(raw) != manifest['source']['samples_json_sha256']:
        raise ValueError('samples.json SHA256 mismatch; use the pinned dataset revision in docs/DATA.md')
    exports = convert_samples(json.loads(raw)['samples'], manifest['classes'])
    for split, export in exports.items():
        expected_hash = manifest['annotation_sha256'][f'instances_{split}.json']
        if sha256(export['data']) != expected_hash:
            raise ValueError(f'{split}: annotation SHA256 differs from the recorded export')
        expected = manifest['splits'][split]
        if len(export['images']) != expected['images'] or export['annotations'] != expected['annotations']:
            raise ValueError(f'{split}: image or annotation count differs from the recorded export')
    return manifest, exports


def check_images(exports: dict, splits: list[str], images_dir: Path) -> None:
    from PIL import Image

    for split in splits:
        for record in exports[split]['images']:
            path = images_dir / record['file_name']
            with Image.open(path) as image:
                if image.size != (record['width'], record['height']):
                    raise ValueError(f'Image size differs from the source metadata: {path}')
                image.verify()


def prepare_dataset(samples_json: Path, output_dir: Path, splits: list[str],
                    images_dir: Path | None = None, image_mode: str = 'copy') -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'Choose a new output directory: {output_dir}')
    if not splits or len(set(splits)) != len(splits) or any(s not in dict(SPLITS) for s in splits):
        raise ValueError('Choose distinct splits from train, val and test-dev')
    if image_mode not in ('copy', 'symlink'):
        raise ValueError('image_mode must be copy or symlink')
    manifest, exports = verified_exports(samples_json)
    if images_dir is not None:
        check_images(exports, splits, images_dir)

    # Keep incomplete outputs for diagnosis if an I/O error occurs after this point.
    output_dir.mkdir(parents=True, exist_ok=False)
    annotations_dir = output_dir / 'annotations'
    annotations_dir.mkdir()
    report = {
        'source_revision': manifest['source']['git_commit'],
        'source_sha256': manifest['source']['samples_json_sha256'],
        'all_annotation_hashes_verified': True,
        'image_checks': 'dimensions_and_readability' if images_dir is not None else 'not_performed',
        'image_mode': image_mode if images_dir is not None else None,
        'splits': {},
    }
    for split in splits:
        export = exports[split]
        (annotations_dir / f'instances_{split}.json').write_bytes(export['data'])
        if images_dir is not None:
            destination = output_dir / 'images' / split
            destination.mkdir(parents=True)
            for record in export['images']:
                source = images_dir / record['file_name']
                target = destination / record['file_name']
                if image_mode == 'symlink':
                    target.symlink_to(source.resolve())
                else:
                    shutil.copy2(source, target)
        report['splits'][split] = {
            'images': len(export['images']), 'annotations': export['annotations'],
            'sha256': sha256(export['data']), 'dropped': export['dropped'],
        }
    (output_dir / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory; never overwritten')
    parser.add_argument('--splits', nargs='+', choices=list(dict(SPLITS)), default=['train', 'val'])
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument('--images-dir', type=Path, help='The pinned dataset snapshot data/ directory')
    image_source.add_argument('--annotations-only', action='store_true', help='Verify/export labels without preparing images')
    parser.add_argument('--image-mode', choices=['copy', 'symlink'], default='copy')
    args = parser.parse_args()
    report = prepare_dataset(args.samples_json, args.output_dir, args.splits,
                             args.images_dir, args.image_mode)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
