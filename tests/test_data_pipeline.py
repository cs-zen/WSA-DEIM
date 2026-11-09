import json
import random
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes

from engine.core import YAMLConfig
from engine.data.transforms import ConvertBoxes, ConvertPILImage, PadToSize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_image_and_box_conversion_values():
    image = Image.new('RGB', (20, 10), (255, 128, 0))
    boxes = BoundingBoxes([[2.0, 1.0, 10.0, 5.0]], format='XYXY', canvas_size=(10, 20))
    pixels = ConvertPILImage()(image)
    converted = ConvertBoxes(fmt='cxcywh', normalize=True)(boxes)
    assert pixels.dtype == torch.float32
    torch.testing.assert_close(pixels[:, 0, 0], torch.tensor([1.0, 128 / 255, 0.0]))
    torch.testing.assert_close(converted, torch.tensor([[0.3, 0.3, 0.4, 0.4]]))


def test_pad_to_size_keeps_boxes_and_records_padding():
    image = Image.new('RGB', (20, 10), (255, 128, 0))
    target = {'boxes': BoundingBoxes([[2.0, 1.0, 10.0, 5.0]], format='XYXY', canvas_size=(10, 20))}
    padded, result = PadToSize((24, 16), fill=7)(image, target)
    assert padded.size == (24, 16)
    assert padded.getpixel((23, 15)) == (7, 7, 7)
    assert result['padding'].tolist() == [0, 0, 4, 6]
    assert result['boxes'].canvas_size == (16, 24)
    torch.testing.assert_close(result['boxes'], target['boxes'])


@pytest.mark.parametrize('epoch', [0, 4, 144, 280])
def test_full_recipe_data_loading_across_augmentation_stages(tmp_path, epoch):
    random.seed(0)
    torch.manual_seed(0)
    images = []
    annotations = []
    for index in range(1, 9):
        Image.new('RGB', (64, 48), (index * 20, 100, 180)).save(tmp_path / f'{index}.png')
        images.append({'id': index, 'file_name': f'{index}.png', 'width': 64, 'height': 48})
        annotations.append({'id': index, 'image_id': index, 'category_id': index,
                            'bbox': [8, 6, 32, 24], 'area': 768, 'iscrowd': 0})
    annotation_path = tmp_path / 'instances.json'
    annotation_path.write_text(json.dumps({
        'images': images, 'annotations': annotations,
        'categories': [{'id': i, 'name': str(i)} for i in range(1, 11)],
    }))
    overrides = {
        split: {'num_workers': 0, 'dataset': {
            'img_folder': str(tmp_path), 'ann_file': str(annotation_path),
        }} for split in ('train_dataloader', 'val_dataloader')
    }
    cfg = YAMLConfig(str(PROJECT_ROOT / 'configs/wsa_deim_visdrone.yml'), **overrides)
    loader = cfg.train_dataloader
    loader.set_epoch(epoch)
    samples, targets = next(iter(loader))
    assert samples.shape == (4, 3, 640, 640)
    assert samples.dtype == torch.float32
    assert torch.isfinite(samples).all()
    assert samples.min() >= 0 and samples.max() <= 1
    for target in targets:
        assert torch.isfinite(target['boxes']).all()
        assert ((target['boxes'] >= 0) & (target['boxes'] <= 1)).all()
        assert ((target['labels'] >= 0) & (target['labels'] < 10)).all()
    val_samples, val_targets = next(iter(cfg.val_dataloader))
    assert val_samples.shape == (8, 3, 640, 640)
    assert len(val_targets) == 8
