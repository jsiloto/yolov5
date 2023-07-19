import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import box_iou, ap_per_class
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--txt', type=str, default=ROOT / 'data/coco128.yaml', help='label results path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt



def main(opt):
    task = 'val'
    pad, rect = (0.0, False) if task == 'speed' else (0.5, True)  # square inference for benchmarks
    task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images

    data, imgsz, batch_size, stride, workers = \
        opt.data, opt.imgsz, opt.batch_size, 32, opt.workers

    data = check_dataset(data)  # check
    print(data['names'])
    dataloader = create_dataloader(data[task],
                                   imgsz,
                                   batch_size,
                                   stride,
                                   False,
                                   pad=pad,
                                   rect=False,
                                   workers=workers,
                                   prefix=colorstr(f'{task}: '))[0]

    device = select_device("cpu", batch_size=batch_size)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar

    stats = []
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        size = im.shape[2:4]
        nb, _, height, width = im.shape  # batch size, channels, height, width
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

        for i in range(len(paths)):
            labelsn = targets[targets[:, 0] == i, 1:]
            pred_file = paths[i].split("/")[-1].split(".")[0] + ".txt"
            pred_file = os.path.join(opt.txt, pred_file)
            try:
                with open(pred_file, "r") as f:
                    # Process detections
                    pb = torch.from_numpy(
                        np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
                    )
                    pb[:, 1:5] *= torch.tensor((width, height, width, height), device=device)

                    #

                    tbox = xywh2xyxy(pb[:, 1:5])
                    # scale_boxes(size, tbox, shapes[i][0], shapes[i][1])
                    predn = torch.cat((tbox, pb[:, 5:6],  pb[:, 0:1]), 1)

                    # Process targets
                    tbox = xywh2xyxy(labelsn[:, 1:5])
                    # scale_boxes(size, tbox, shapes[i][0], shapes[i][1])
                    labelsn = torch.cat((labelsn[:, 0:1], tbox), 1)
                    correct = process_batch(predn, labelsn, iouv)
                    stats.append((correct, predn[:, 4], predn[:, 5], labelsn[:, 0]))
            except FileNotFoundError:
                print(f"FileNotFoundError: {pred_file}")
                continue

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="./", names=data['names'])
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        print(mp, mr, map50, map)
    exit()

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
