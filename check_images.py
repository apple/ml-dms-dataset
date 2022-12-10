#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to use the rotation descriptor and verify images.
#

import argparse
import json
import gzip
import os
import cv2
import numpy as np
import PIL.Image as Image


def rotation_descriptor(ipath):
    # this calculates the descriptor
    orig_img = cv2.imread(ipath).mean(axis=2, dtype=np.float32)
    img = cv2.resize(orig_img, (90, 90), interpolation=cv2.INTER_AREA)
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    dx_grid = dx.reshape(3, 30, 3, 30).mean(axis=(1, 3))
    dy_grid = dy.reshape(3, 30, 3, 30).mean(axis=(1, 3))
    angles = np.rad2deg(np.arctan2(dy_grid, dx_grid)) % 360
    return orig_img.shape, angles


def angle_subtract(a, b, deg=False):
    # this converts polar to cartesian to find a polar
    # coordinate in the range of [-pi, pi] with atan2
    amb = a - b
    if deg:
        amb = np.deg2rad(amb)
    result = np.arctan2(np.sin(amb), np.cos(amb))
    if deg:
        result = np.rad2deg(result)
    return result


def rotation_descriptor_match(a, b, percentile, tolerance, max_tolerance):
    # a matches b if percentile elements are within the tolerance and
    # no element is beyond the maximum tolerance.
    absdiff = np.abs(angle_subtract(a, b, deg=True))
    m1 = (absdiff <= tolerance).mean() >= percentile / 100
    m2 = np.all(absdiff <= max_tolerance)
    return m1 and m2


def main(args):
    data = json.loads(
        gzip.open(os.path.join(args.data_path, 'info.json.gz'), 'rb').read()
    )
    print(f'Dataset path is {args.data_path}.')
    print(f'Dataset describes {len(data)} images.')
    issues = {
        'image not found': [],
        'image found, label not found': [],
        'incorrect size': [],
        'rotation may not match': [],
        'PIL and cv2 size disagreement': [],
    }
    passed = 0
    for datum in data:
        p1 = os.path.join(args.data_path, datum['image_path'])
        p2 = os.path.join(args.data_path, datum['label_path'])
        # image exists?
        if not os.path.exists(p1):
            issues['image not found'].append(p1)
            continue
        # label exists?
        if not os.path.exists(p2):
            issues['image found, label not found'].append(p1)
            continue
        # size match?
        shape, rd = rotation_descriptor(p1)
        size = Image.open(p2).size
        if shape[0] != size[1] or shape[1] != size[0]:
            issues['incorrect size'].append(p1)
            continue
        # rotation match?
        warning = False
        ref = np.asarray(datum['rotation_descriptor'])
        if not rotation_descriptor_match(rd, ref, 50, 5, 30):
            issues['rotation may not match'].append(p1)
            warning = True
        # inconsistent size?
        size = Image.open(p1).size
        if shape[0] != size[1] or shape[1] != size[0]:
            issues['PIL and cv2 size disagreement'].append(p1)
            warning = True
        if not warning:
            passed += 1
    print(f'{passed} passed with no issues.')
    print('Summary of issues found:')
    for k, v in issues.items():
        print(f'  {k}: {len(v)}')
    print('See image_issues.json for details.')
    issues[
        'instructions'
    ] = 'An incorrect size is resolved by resizing the image to match the label map. Potential rotation mismatches are resolved by rotating the image so its orientation matches the label map. A size disagreement between PIL and cv2 is likely due to EXIF tags, which is resolved by stripping EXIF from the image and rotating the image so its orientation matches the label map.'
    open('image_issues.json', 'w').write(json.dumps(issues, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to DMS dataset'
    )
    args = parser.parse_args()
    main(args)
