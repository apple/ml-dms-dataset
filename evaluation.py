#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to evaluate dataset images.
#

import argparse
import torchvision.transforms as TTR
import os
import random
import json
import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from torch.utils.data import DataLoader
from torch.utils import data

random.seed(112)
ignore_index = 255
dms46 = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 56, ]
t = json.load(open(os.path.expanduser('./taxonomy.json'), 'rb'))
srgb_colormap = [
    t['srgb_colormap'][i] for i in range(len(t['srgb_colormap'])) if i in dms46
]
srgb_colormap.append([0, 0, 0])  # color for ignored index
srgb_colormap = np.array(srgb_colormap, dtype=np.uint8)


class DatasetReader(data.Dataset):
    def __init__(self, args, split):
        print(f'======={split}:DMS==================')
        self.data_path = args.data_path
        self.num_classes = args.num_classes
        assert self.num_classes == 46

        self.args = args
        self.split = split

        self.images_path = os.path.join(self.data_path, f'images/{self.split}')
        self.anno_path = os.path.join(self.data_path, f'labels/{self.split}')
        data = [i for i in os.listdir(self.images_path) if '.jpg' in i]
        data = [x for x in data if os.path.exists(self.segmap_path(x))]
        self.data = data
        assert len(self.data) > 0, f'No image and label pairs found, check that data_path is correct'

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        self.mean = mean
        self.std = std

        self.setup_class_mapping()

        print(f'valid_classes {self.valid_classes}')
        print(f'merge_dict {self.merge_dict}')
        print(f'class_map {self.class_map}')

        print(f'{self.split} has this many images: {len(self.data)}')

    def setup_class_mapping(self):
        # filter classes not predicted
        taxonomy_file = json.load(open(os.path.expanduser('./taxonomy.json'), 'rb'))
        self.all_names = taxonomy_file['names']
        self.valid_classes = dms46
        self.valid_classes.sort()
        assert (
            len(self.valid_classes) == self.num_classes
        ), 'valid_classes doesnt match num_classes'
        self.merge_dict = {
            0: set([i for i in range(len(self.all_names))]) ^ set(self.valid_classes)
        }
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

    def segmap_path(self, filename):
        return os.path.join(self.anno_path, filename[:-4] + '.png')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]
        img_path = os.path.join(self.images_path, filename)
        seg_path = self.segmap_path(filename)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        assert len(seg.shape) == 2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert (
            img.shape[:2] == seg.shape
        ), f'Shape mismatch for {img_path}, {img.shape[:2]} vs. {seg.shape}'
        label = self.encode_segmap(seg)

        image = np.copy(img)

        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        label = torch.from_numpy(label).long()
        image = TTR.Normalize(self.mean, self.std)(image)

        return (
            image,
            {
                'original_image': torch.from_numpy(np.uint8(img).copy()),
                'target': label,
                'filename': filename,
            },
        )

    def encode_segmap(self, mask):
        # combine categories
        for i, j in self.merge_dict.items():
            for k in j:
                mask[mask == k] = i

        # assign ignored index
        mask[mask == 0] = ignore_index

        # map valid classes to id
        # use valid_classes sorted so will not remap.
        for valid_class in self.valid_classes:
            assert valid_class > self.class_map[valid_class]
            mask[mask == valid_class] = self.class_map[valid_class]

        return mask


def apply_color(label_mask):
    # translate labels to visualization colors
    label_mask[label_mask == ignore_index] = len(srgb_colormap) - 1
    vis = np.take(srgb_colormap, label_mask, axis=0)
    return vis[..., ::-1]


def main(args):
    is_cuda = torch.cuda.is_available()
    model = torch.jit.load(args.jit_path)
    if is_cuda:
        model = model.cuda()

    os.makedirs(args.output_folder, exist_ok=True)
    val_set = DatasetReader(args, split='validation')
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    confusion_matrix = np.zeros((46, 46))
    for i, (input_image, target) in enumerate(val_loader):
        if i % 200 == 0:
            print(f'Testing-: ', i)
        if is_cuda:
            input_image = input_image.cuda()

        with torch.no_grad():
            prediction = model(input_image)[0].data.cpu()[0].numpy()

        gt = target['target'].cpu().numpy()
        mask = gt != ignore_index

        confusion_matrix += sklearn_confusion_matrix(
            gt[mask], prediction[mask], labels=[_ for _ in range(46)]
        )

        if args.visualize and i < 20:
            original_image = target['original_image'][0].cpu().numpy()[..., ::-1]
            target_colored = apply_color(gt[0])
            predicted_colored = apply_color(prediction[0])

            stacked_img = np.concatenate(
                (np.uint8(original_image), target_colored, predicted_colored), axis=1
            )
            cv2.imwrite(
                f'{args.output_folder}/{os.path.basename(target["filename"][0])[:-4]}.png',
                stacked_img,
            )

    pixel_acc = np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-9)
    per_class_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + 1e-9)
    macc = per_class_acc.sum() / ((np.sum(confusion_matrix, axis=1) > 0).sum() + 1e-9)
    per_class_iou = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1)
        + np.sum(confusion_matrix, axis=0)
        - np.diag(confusion_matrix)
        + 1e-9
    )
    miou = per_class_iou.sum() / ((np.sum(confusion_matrix, axis=1) > 0).sum() + 1e-9)

    class_names = [val_set.all_names[i] for i in val_set.valid_classes]
    results = {
        'class names': class_names,
        'pixel accuracy': pixel_acc,
        'mean class accuracy': macc,
        'mean IOU': miou,
        'per-class accuracy': per_class_acc.tolist(),
        'per-class IOU': per_class_iou.tolist(),
    }
    print('Results:')
    for k, v in results.items():
        print(f'  {k}: {v}')
    print('See evaluation_results.json for details.')
    open('evaluation_results.json', 'w').write(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jit_path',
        type=str,
        required=True,
        help='path to the pretrained model',
    )

    parser.add_argument('--visualize', type=bool, default=True, help='visualize or not')

    parser.add_argument(
        '--data_path', type=str, required=True, help='path to DMS dataset'
    )
    parser.add_argument(
        '--output_folder', type=str, default='.', help='where to save visualization'
    )
    parser.add_argument(
        '--num_classes', type=int, default=46, help='number of classes to predict'
    )

    args = parser.parse_args()
    main(args)
