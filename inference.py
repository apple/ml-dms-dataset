#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to predict materials.
#

import argparse
import torchvision.transforms as TTR
import os
import glob
import random
import json
import cv2
import numpy as np
import torch
import math
from PIL import Image

random.seed(112)

dms46 = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 56, ]
t = json.load(open(os.path.expanduser('./taxonomy.json'), 'rb'))
srgb_colormap = [
    t['srgb_colormap'][i] for i in range(len(t['srgb_colormap'])) if i in dms46
]
srgb_colormap = np.array(srgb_colormap, dtype=np.uint8)


def apply_color(label_mask):
    # translate labels to visualization colors
    vis = np.take(srgb_colormap, label_mask, axis=0)
    return vis[..., ::-1]


def main(args):
    is_cuda = torch.cuda.is_available()
    model = torch.jit.load(args.jit_path)
    if is_cuda:
        model = model.cuda()

    images_list = glob.glob(f'{args.image_folder}/*')
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    os.makedirs(args.output_folder, exist_ok=True)

    for image_path in images_list:
        print(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_dim = 512
        h, w = img.shape[0:2]
        scale_x = float(new_dim) / float(h)
        scale_y = float(new_dim) / float(w)
        scale = min(scale_x, scale_y)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
        img = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
        img = np.array(img)

        image = torch.from_numpy(img.transpose((2, 0, 1))).float()
        image = TTR.Normalize(mean, std)(image)
        if is_cuda:
            image = image.cuda()
        image = image.unsqueeze(0)
        with torch.no_grad():
            prediction = model(image)[0].data.cpu()[0, 0].numpy()
        original_image = img[..., ::-1]

        predicted_colored = apply_color(prediction)

        stacked_img = np.concatenate(
            (np.uint8(original_image), predicted_colored), axis=1
        )
        cv2.imwrite(
            f'{args.output_folder}/{os.path.splitext(os.path.basename(image_path))[0]}.png',
            stacked_img,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jit_path',
        type=str,
        default='',
        help='path to the pretrained model',
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        default='',
        help='overwrite the data_path for local runs',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='',
        help='overwrite the data_path for local runs',
    )
    args = parser.parse_args()
    main(args)
