#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to resize the original images.
#

import argparse
import json
import gzip
import os
import cv2
import numpy as np
import shutil
import urllib.parse
import posixpath
import PIL.Image as Image


def ensuredir(x):
    a, b = os.path.split(x)
    if not a:
        return
    if os.path.exists(a):
        return
    if not os.path.exists(a):
        ensuredir(a)
    os.mkdir(a)


def main(args):
    data = json.loads(
        gzip.open(os.path.join(args.data_path, 'info.json.gz'), 'rb').read()
    )
    print(f'Dataset path is {args.data_path}.')
    print(f'Dataset describes {len(data)} images.')
    print(f'Original images are in {args.originals_path}.')
    outcomes = {
        'prepared image found': [],
        'original image not found': [],
        'image copied': [],
        'image resized': [],
        'image not processed': [],
    }
    for datum in data:
        p1 = os.path.join(args.data_path, datum['image_path'])
        if os.path.exists(p1):
            outcomes['prepared image found'].append(p1)
            continue
        url = datum['openimages_metadata']['OriginalURL']
        original_name = posixpath.split(urllib.parse.urlparse(url).path)[1]
        p2 = os.path.join(args.originals_path, original_name)
        if not os.path.exists(p2):
            outcomes['original image not found'].append(p1)
            continue
        original = cv2.imread(p2)
        if original.shape[0] == datum['height'] and original.shape[1] == datum['width']:
            # copy
            ensuredir(p1)
            shutil.copy(p2, p1)
            outcomes['image copied'].append(p1)
        else:
            # try to resize
            scale_factor = datum['height'] / original.shape[0]
            alt_scale_factor = datum['width'] / original.shape[1]
            if np.allclose(scale_factor, alt_scale_factor, rtol=0.01):
                # resize
                new_size = (datum['width'], datum['height'])
                method = cv2.INTER_AREA if scale_factor <= 1 else cv2.INTER_CUBIC
                result = cv2.resize(original, new_size, interpolation=method)
                ensuredir(p1)
                Image.fromarray(result[:, :, ::-1]).save(p1, quality=95)
                assert os.path.exists(p1)
                outcomes['image resized'].append(p1)
            else:
                # do not distort the aspect ratio more than 1%
                outcomes['image not processed'].append(p1)
    print('Summary of preparation:')
    for k, v in outcomes.items():
        print(f'  {k}: {len(v)}')
    print('See preparation_outcomes.json for details.')
    outcomes[
        'documentation'
    ] = 'An image is not processed if resizing it would cause the aspect ratio to change by more than 1%. These images should be manually resized and checked for distortion. After preparing the images it is important to run check_images.py, which checks if image rotation matches the labels.'
    open('preparation_outcomes.json', 'w').write(json.dumps(outcomes, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to DMS dataset'
    )
    parser.add_argument(
        '--originals_path',
        type=str,
        required=True,
        help='path to directory of original images',
    )
    args = parser.parse_args()
    main(args)
