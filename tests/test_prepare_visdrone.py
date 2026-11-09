import json
from pathlib import Path

import pytest
from PIL import Image

from tools import prepare_visdrone as prepare


CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
           'tricycle', 'awning-tricycle', 'bus', 'motor']


def sample(name, tag, label, bbox):
    return {
        'filepath': f'data/{name}.jpg', 'tags': [tag],
        'metadata': {'width': 100, 'height': 50},
        'ground_truth': {'detections': [{
            'label': label, 'bounding_box': bbox, 'truncation': 1, 'occlusion': 2,
        }]},
    }


def test_recorded_conversion_semantics():
    samples = [
        sample('z', 'train', 'awning_tricycle', [0.2, 0.2, 0.1, 0.2]),
        sample('a', 'train', 'pedestrians', [0.1, 0.2, 0.3, 0.4]),
        sample('b', 'val', 'ignore_regions', [0, 0, 1, 1]),
        sample('c', 'val', 'others', [0, 0, 1, 1]),
        sample('d', 'val', 'car', [0, 0, 0, 0.5]),
        sample('e', 'val', 'car', [-0.1, 0, 0.2, 0.4]),
        sample('f', 'test', 'bus', [0.2, 0.2, 0.1, 0.2]),
    ]
    exports = prepare.convert_samples(samples, CLASSES)
    train = json.loads(exports['train']['data'])
    assert [im['file_name'] for im in train['images']] == ['a.jpg', 'z.jpg']
    assert train['annotations'][0] == {
        'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10.0, 10.0, 30.0, 20.0],
        'area': 600.0, 'iscrowd': 0, 'segmentation': [], 'truncation': 1, 'occlusion': 2,
    }
    assert train['annotations'][1]['category_id'] == 8
    val = json.loads(exports['val']['data'])
    assert len(val['images']) == 4  # Do not discard empty images.
    assert val['annotations'][0]['id'] == 3  # IDs continue across splits.
    assert val['annotations'][0]['bbox'] == [-10.0, 0, 20.0, 20.0]  # No export-time clipping.
    assert exports['val']['dropped'] == {'ignore_regions': 1, 'others': 1, 'degenerate_boxes': 1}
    test = json.loads(exports['test-dev']['data'])
    assert test['annotations'][0]['id'] == 4
    assert test['images'][0]['id'] == 1


def test_unverified_source_is_rejected_without_output(tmp_path):
    source = tmp_path / 'samples.json'
    source.write_text('{"samples": []}')
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        prepare.prepare_dataset(source, output, ['train'])
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    sentinel = tmp_path / 'keep.txt'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        prepare.prepare_dataset(tmp_path / 'missing.json', tmp_path, ['train'])
    assert sentinel.read_text() == 'keep'


@pytest.mark.parametrize('mode', ['copy', 'symlink'])
def test_image_layout_and_size_checks(tmp_path, monkeypatch, mode):
    exports = prepare.convert_samples([sample('a', 'val', 'car', [0, 0, 0.2, 0.2])], CLASSES)
    monkeypatch.setattr(prepare, 'verified_exports', lambda _: (
        {'source': {'git_commit': 'test', 'samples_json_sha256': 'test'}}, exports))
    images = tmp_path / 'images'
    images.mkdir()
    image_path = images / 'a.jpg'
    Image.new('RGB', (99, 50)).save(image_path)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='Image size differs'):
        prepare.prepare_dataset(Path('unused'), output, ['val'], images, mode)
    assert not output.exists()
    Image.new('RGB', (100, 50)).save(image_path)
    report = prepare.prepare_dataset(Path('unused'), output, ['val'], images, mode)
    destination = output / 'images/val/a.jpg'
    assert destination.read_bytes() == image_path.read_bytes()
    assert destination.is_symlink() == (mode == 'symlink')
    assert report['splits']['val']['images'] == 1
    assert (output / 'annotations/instances_val.json').read_bytes() == exports['val']['data']
