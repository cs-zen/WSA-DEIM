# Modified for the WSA-DEIM full-model code release (2026).
"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
import os
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _iter_trace_tensors(value, path):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_trace_tensors(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_trace_tensors(item, f'{path}[{index}]')


def _nonfinite_tensor_summary(value):
    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    if bool(finite_mask.all()):
        return None

    finite_values = detached[finite_mask]
    summary = {
        'shape': tuple(detached.shape),
        'dtype': str(detached.dtype),
        'nan': int(torch.isnan(detached).sum().item()),
        'posinf': int(torch.isposinf(detached).sum().item()),
        'neginf': int(torch.isneginf(detached).sum().item()),
    }
    if finite_values.numel() > 0:
        summary['finite_min'] = float(finite_values.min().item())
        summary['finite_max'] = float(finite_values.max().item())
    return summary


def _install_nonfinite_trace(model, context):
    if os.environ.get('DEIM_TRACE_NONFINITE') != '1':
        return []

    traced_classes = {
        'FSFF',
        'GLCAM',
        'SNI',
        'Haar_Spatial_Alignment',
        'ConvNormLayer_fuse',
        'UpsampleFusion',
        'DownsampleFusion',
        'HaarFPN',
        'HaarPAN',
        'HybridEncoder',
    }

    handles = []
    traced_names = []

    def make_hook(module_name):
        def trace_hook(_module, inputs, output):
            for output_path, tensor in _iter_trace_tensors(output, 'output'):
                summary = _nonfinite_tensor_summary(tensor)
                if summary is None:
                    continue

                input_summaries = []
                for input_path, input_tensor in _iter_trace_tensors(inputs, 'input'):
                    input_summary = _nonfinite_tensor_summary(input_tensor)
                    if input_summary is not None:
                        input_summaries.append((input_path, input_summary))

                print(
                    'DEIM_NONFINITE_TRACE '
                    f'module={module_name} tensor={output_path} '
                    f'summary={summary} nonfinite_inputs={input_summaries} '
                    f'context={context}',
                    flush=True,
                )
                raise FloatingPointError(
                    f'non-finite encoder activation at {module_name}:{output_path}'
                )
        return trace_hook

    for name, module in model.named_modules():
        normalized_name = name.removeprefix('module.')
        if not normalized_name.startswith('encoder'):
            continue
        if module.__class__.__name__ not in traced_classes:
            continue
        handles.append(module.register_forward_hook(make_hook(normalized_name)))
        traced_names.append(normalized_name)

    print(
        f'DEIM_NONFINITE_TRACE enabled modules={traced_names}',
        flush=True,
    )
    return handles


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    trace_context = {'epoch': epoch}
    trace_handles = _install_nonfinite_trace(model, trace_context)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
        trace_context.update(
            step=i,
            global_step=global_step,
            image_ids=[
                int(target['image_id'].reshape(-1)[0].item())
                for target in targets
                if 'image_id' in target and target['image_id'].numel() > 0
            ],
        )

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace('module.', '')
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    for handle in trace_handles:
        handle.remove()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = coco_evaluator.iou_types

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)


        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
