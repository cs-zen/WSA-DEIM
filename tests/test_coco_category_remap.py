import json
from pathlib import Path

import torch
from PIL import Image

from engine.core.yaml_utils import load_config
from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.dataset.coco_eval import CocoEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_visdrone_recipe_keeps_image_transforms():
    config = load_config(
        str(PROJECT_ROOT / 'configs/wsa_deim_visdrone.yml'),
        cfg={},
    )

    train_ops = config['train_dataloader']['dataset']['transforms']['ops']
    val_ops = config['val_dataloader']['dataset']['transforms']['ops']

    assert [op['type'] for op in train_ops][-3:] == [
        'SanitizeBoundingBoxes',
        'ConvertPILImage',
        'ConvertBoxes',
    ]
    assert [op['type'] for op in val_ops] == ['Resize', 'ConvertPILImage']


def test_visdrone_scratch_recipe_matches_300_epoch_experiment():
    config = load_config(
        str(
            PROJECT_ROOT
            / 'configs/wsa_deim_visdrone.yml'
        ),
        cfg={},
    )

    transforms = config['train_dataloader']['dataset']['transforms']
    policy = transforms['policy']
    collate = config['train_dataloader']['collate_fn']

    assert config['HGNetv2']['pretrained'] is False
    assert config['seed'] == 0
    assert config['epoches'] == 300
    assert config['checkpoint_freq'] == 30
    assert config['flat_epoch'] == 144
    assert config['no_aug_epoch'] == 20
    assert config['train_dataloader']['total_batch_size'] == 4
    assert config['val_dataloader']['total_batch_size'] == 8
    assert policy['epoch'] == [4, 144, 280]
    assert transforms['mosaic_prob'] == 0.5
    assert collate['base_size'] == 640
    assert collate['base_size_repeat'] is None
    assert collate['mixup_prob'] == 0.5
    assert collate['mixup_epochs'] == [4, 144]
    assert collate['stop_epoch'] == 280


def test_non_contiguous_coco_category_ids_are_remapped(tmp_path):
    image_path = tmp_path / 'sample.jpg'
    Image.new('RGB', (32, 24)).save(image_path)

    annotation_path = tmp_path / 'instances.json'
    annotation_path.write_text(
        json.dumps(
            {
                'images': [
                    {'id': 1, 'file_name': image_path.name, 'width': 32, 'height': 24}
                ],
                'annotations': [
                    {
                        'id': 1,
                        'image_id': 1,
                        'category_id': 10,
                        'bbox': [1, 2, 10, 8],
                        'area': 80,
                        'iscrowd': 0,
                    }
                ],
                'categories': [
                    {'id': 10, 'name': 'motor'},
                    {'id': 1, 'name': 'pedestrian'},
                ],
            }
        ),
        encoding='utf-8',
    )

    dataset = CocoDetection(
        img_folder=str(tmp_path),
        ann_file=str(annotation_path),
        transforms=None,
        remap_category_ids=True,
    )
    _, target = dataset.load_item(0)

    assert target['labels'].tolist() == [1]
    assert dataset.category2label == {1: 0, 10: 1}
    assert dataset.label2category == {0: 1, 1: 10}

    evaluator = CocoEvaluator(
        dataset.coco,
        iou_types=['bbox'],
        remap_category_ids=True,
    )
    results = evaluator.prepare_for_coco_detection(
        {
            1: {
                'boxes': torch.tensor([[1.0, 2.0, 11.0, 10.0]]),
                'scores': torch.tensor([0.9]),
                'labels': torch.tensor([1]),
            }
        }
    )

    assert results[0]['category_id'] == 10
