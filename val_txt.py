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
from decidermodel import DeciderModel

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'yolov5/data/bdd100kvideo.yaml', help='dataset.yaml path')
    parser.add_argument('--txt1', type=str, default=ROOT / 'movie_dataset/daytime/val/labels',
                        help='label results path')
    parser.add_argument('--txt2', type=str, default=ROOT / 'movie_dataset/daytime/val/labels',
                        help='label results path')
    parser.add_argument('--save', type=str, default="", help='Save Model dataset')
    parser.add_argument('--task', type=str, default="val", help="val or test or train")
    parser.add_argument('--weights', type=str, default="", help="val or test or train")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt


def txtval(data, txt1, txt2, save=False, task='val', weights=""):
    pad, rect = (0.0, False) if task == 'speed' else (0.5, True)  # square inference for benchmarks
    task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    batch_size = 32
    imgsz = 640
    workers = 8
    stride = 32
    decider = None

    data = check_dataset(data)  # check
    print(data['names'])
    dataloader = create_dataloader(data[task],
                                   imgsz,
                                   batch_size,
                                   stride,
                                   False,
                                   pad=pad,
                                   rect=False,
                                   shuffle=True,
                                   workers=workers,
                                   prefix=colorstr(f'{task}: '))[0]

    device = select_device("cpu", batch_size=batch_size)

    print(weights)
    if len(weights) > 0:
        decider = DeciderModel().to(device)
        decider.load_state_dict(torch.load(weights, map_location=device))

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95

    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    stats1 = []
    stats2 = []
    stats_best = []
    map_dataset = {}
    correct = 0
    total_images = 0
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        nb, _, height, width = im.shape  # batch size, channels, height, width
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

        for i in range(len(paths)):
            total_images += 1
            labelsn = targets[targets[:, 0] == i, 1:]

            pred_file = paths[i].split("/")[-1].split(".")[0] + ".txt"
            pred_file1 = os.path.join(txt1, pred_file)
            pred_file2 = os.path.join(txt2, pred_file)
            try:

                # Process detections
                single_stats1 = get_single_stats(pred_file1, labelsn, iouv, width, height, device)
                single_stats2 = get_single_stats(pred_file2, labelsn, iouv, width, height, device)
                                                 # ,ignore_classes=[0, 2, 7, 8])
                ap1 = single_img_ap(single_stats1, data['names'])
                ap2 = single_img_ap(single_stats2, data['names'])

                # # print(labelsn[:,0].int())
                # for c in [4, 5, 6]:
                #     index = labelsn[:,0].int() == c
                #     if index.any():
                #         ap2 = 0.0

                if ap1 == ap2 == 0.0:
                    map_dataset[pred_file] = [0.01, 0.01]
                else:
                    map_dataset[pred_file] = [ap1, ap2]

                stats1 += single_stats1
                stats2 += single_stats2
                if decider is not None:
                    with torch.no_grad():
                        inputs = im[i].to(device, non_blocking=True).float() / 255
                        out = decider(inputs.unsqueeze(0))
                        out = out > 0.5
                        out = out.item()
                        correct = correct + 1 if out == (ap2 > ap1) else correct
                        if out:
                            stats_best += single_stats2
                        else:
                            stats_best += single_stats1
                else:
                    stats_best += single_stats1 if ap1 > ap2 else single_stats2

            except FileNotFoundError:
                map_dataset[pred_file] = [0.01, 0.01]
                print(f"FileNotFoundError: {pred_file}")
                continue

    if len(save):
        assert save.endswith(".json"), "save file must be json"
        with open(save, "w") as f:
            json.dump(map_dataset, f)

    print(f"Acc: {correct / total_images}")
    map1 = final_map(stats1, data)
    map2 = final_map(stats2, data)
    map_best = final_map(stats_best, data)
    return map1, map2, map_best


def final_map(stats, data):
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="./", names=data['names'])
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        print(mp, mr, map50, map)
    return map


def get_single_stats(pred_file, labelsn, iouv, width, height, device, ignore_classes=None):
    with open(pred_file, "r") as f:
        pb = torch.from_numpy(
            np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        )
    pb[:, 1:5] *= torch.tensor((width, height, width, height), device=device)

    tbox = xywh2xyxy(pb[:, 1:5])
    # scale_boxes(size, tbox, shapes[i][0], shapes[i][1])
    predn = torch.cat((tbox, pb[:, 5:6], pb[:, 0:1]), 1)
    if ignore_classes is not None:
        for c in ignore_classes:
            index = predn[:, 5] == c
            if index.any():
                predn = predn[index]
                # print(index)

    #
    # cc = [c in predn.T[-1].int() for c in [4,5,6]]
    # cc = cc[0] or cc[1] or cc[2]
    # print(cc)

    # Process targets
    tbox = xywh2xyxy(labelsn[:, 1:5])
    # scale_boxes(size, tbox, shapes[i][0], shapes[i][1])
    labelsn = torch.cat((labelsn[:, 0:1], tbox), 1)

    correct = process_batch(predn, labelsn, iouv)
    single_stats = [(correct, predn[:, 4], predn[:, 5], labelsn[:, 0])]
    return single_stats


def single_img_ap(single_stats, names):
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*single_stats)]  # to numpy
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="./",
                                                  names=names)

    unique_classes, nt = np.unique(single_stats[0][3], return_counts=True)
    classes = list(range(9))
    nt2 = [0] * len(classes)
    for i in range(len(unique_classes)):
        nt2[int(unique_classes[i])] = nt[i]

    nt = np.array(nt2)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    if sum(nt) == 0:
        return 0
    return sum(ap * nt / sum(nt))


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
    txtval(opt.data, opt.txt1, opt.txt2, opt.save, opt.task, opt.weights)
