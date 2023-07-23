# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torchvision

from decidermodel import DeciderModel

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

import torch.nn as nn
import torch.nn.functional as F




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'yolov5s.pt', help='Context1 model')
    parser.add_argument('--train-labels', type=str, default="./train.json", help='Train labels')
    parser.add_argument('--val-labels', type=str, default="./val.json", help='Val labels')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return parser.parse_args()


def main(opt):
    pad, rect = (0.5, True)  # square inference for benchmarks
    batch_size = 16
    imgsz = 640
    workers = 8
    stride = 32
    data = check_dataset(opt.data)  # check
    print(data['names'])

    with open(opt.train_labels) as f:
        train_labels = json.load(f)
    train_dataloader = create_dataloader(data['train'],
                                         imgsz,
                                         batch_size,
                                         stride,
                                         False,
                                         pad=pad,
                                         rect=False,
                                         workers=workers,
                                         shuffle=True,
                                         prefix=colorstr(f'train'))[0]

    with open(opt.val_labels) as f:
        val_labels = json.load(f)
    val_dataloader = create_dataloader(data['val'],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       False,
                                       pad=pad,
                                       rect=False,
                                       shuffle=True,
                                       workers=workers,
                                       prefix=colorstr(f'val'))[0]

    #################################################################################
    device = torch.device('cuda')
    # Model
    model = DeciderModel()
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # Loss
    criterion = nn.BCELoss()

    for epoch in range(30):
        #################################################################################
        # Train Loop
        model.train()
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        pbar = tqdm(train_dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        total_loss = 0
        for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            labelsb = torch.tensor([train_labels[p.split(".jpeg")[0].split("/")[-1]+".txt"] for p in paths]).cuda()
            labels = torch.argmax(labelsb, dim=1).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if loss.isnan():
                print("loss is nan")
                print(labelsb)
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_items = batch_i+1
            # scheduler.step()
            pbar.set_description(f'Epoch {epoch} | train loss: {total_loss/num_items:.4f}')

        #################################################################################
        # Val Loop
        model.eval()
        pbar = tqdm(val_dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        total_loss = 0
        best_acc = 0
        total_positive = 0
        num_items = 0
        with torch.no_grad():
            for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
                labelsb = torch.tensor([val_labels[p.split(".jpeg")[0].split("/")[-1]+".txt"] for p in paths]).cuda()
                labels = torch.argmax(labelsb, dim=1).unsqueeze(1).float()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                ones = outputs > 0.5
                num_items += len(ones)
                total_positive += torch.sum(ones==labels).item()
                acc = total_positive/num_items
                pbar.set_description(f'Epoch {epoch} | '
                                     f'val loss: {total_loss/(batch_i+1):.4f} |'
                                     f' acc = {acc:.4f}' )

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "best.pt")

    #save model
    torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
