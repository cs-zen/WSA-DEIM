import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageOps

from engine.core import YAMLConfig


def load_model(config, checkpoint, device):
    cfg = YAMLConfig(str(config))
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        state = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint
    model = cfg.model
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().deploy()
    postprocessor = cfg.postprocessor.to(device).eval()
    return model, postprocessor, cfg.yaml_cfg['eval_spatial_size']


@torch.inference_mode()
def predict(image, model, postprocessor, size, device, threshold):
    tensor = TF.to_tensor(TF.resize(image, size)).unsqueeze(0).to(device)
    original_size = torch.tensor([image.size], device=device)
    result = postprocessor(model(tensor), original_size)[0]
    selected = result['scores'] >= threshold
    return {key: value[selected].cpu() for key, value in result.items()}


def annotate(image, result):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(result['labels'], result['boxes'], result['scores']):
        coordinates = box.tolist()
        draw.rectangle(coordinates, outline='red', width=2)
        draw.text(coordinates[:2], f'{int(label)} {float(score):.2f}', fill='red')
    return image


def main(args):
    source, destination = Path(args.input), Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f'Choose a new output path: {destination}')
    if args.json_output and Path(args.json_output).exists():
        raise FileExistsError(args.json_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model, postprocessor, size = load_model(args.config, args.checkpoint, args.device)
    if source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
        result = predict(image, model, postprocessor, size, args.device, args.threshold)
        annotate(image, result).save(destination)
        if args.json_output:
            path = Path(args.json_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({k: v.tolist() for k, v in result.items()}, indent=2) + '\n')
    else:
        if args.json_output:
            raise ValueError('--json-output is supported for images only')
        if destination.suffix.lower() != '.mp4':
            raise ValueError('Video output must end in .mp4')
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError('For video, install requirements-video.txt') from error
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f'Cannot open video: {source}')
        writer = None
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not 0 < fps < 1000:
                raise ValueError(f'Invalid source frame rate: {fps}')
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot create video: {destination}')
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = predict(image, model, postprocessor, size, args.device, args.threshold)
                frame = cv2.cvtColor(np.asarray(annotate(image, result)), cv2.COLOR_RGB2BGR)
                writer.write(frame)
            if writer is None:
                raise ValueError('Video contains no readable frames')
        finally:
            capture.release()
            if writer is not None:
                writer.release()
    print(f'Saved {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run WSA-DEIM on an image or video with a compatible local checkpoint.')
    parser.add_argument('-c', '--config', default='configs/wsa_deim_visdrone.yml')
    parser.add_argument('-r', '--checkpoint', required=True)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--json-output', help='Optional image detections, with zero-based training labels')
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    main(args)
