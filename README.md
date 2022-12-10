# The Dense Material Segmentation Dataset

The Dense Material Segmentation Dataset (DMS) consists of 3 million polygon labels of material categories (metal, wood, glass, etc) for 44 thousand RGB images. The dataset is described in the research paper, [A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing](https://arxiv.org/abs/2207.10614).

## Downloading

Our data consists of annotations in the form of label maps. The corresponding RGB images are part of [Open Images](https://storage.googleapis.com/openimages/web/index.html) and must be acquired separately.

This archive contains the fused label maps which are the majority label across multiple annotators. [Download](https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms_v1_labels.zip) (630 MB). This is the only archive needed to train and evaluate material segmentation models.

This archive contains a pre-trained model which predicts 46 kinds of materials. [Download](https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms46_v1.zip) (170 MB).

This archive contains the polygon annotations which were used to create the fused labels. [Download](https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms_v1_polygons.zip) (2.3 GB).

## Data License

The dataset is licensed for non-commercial use under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The pre-trained model is licensed under [LICENSE](LICENSE.txt).

Licenses for the original RGB images must be acquired separately. See [Open Images](https://storage.googleapis.com/openimages/web/index.html) for further information.

## Citation

If you find our data useful in your research, please cite

```
@inproceedings{dmsdataset,
    author    = {Upchurch, Paul and Niu, Ransen},
    title     = {A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2022}
}
```


# Sample Code

## Python Environment

The sample code requires a GPU. The package requirements are described in `environment.yaml`.

```
conda env create -f environment.yaml
conda activate ml-dms
```

## Preparing the Color Images

Step 1. Download or copy the original images into a single directory. Then run the script below, which resizes the images and places them inside `DMS_v1/images`.

```
python prepare_images.py --data_path DMS_v1 --originals_path path/to/originals
```

Step 2. Run the script below, which checks each image and label pair for consistency. The script will also check the image rotation and report if any images need to be manually resized. The script also reports images which are treated differently by software libraries (likely due to relying on an EXIF rotation tag). The generated `image_issues.json` includes instructions for addressing each issue.

```
python check_images.py --data_path DMS_v1
```

## Dataset Evaluation

Step 1. Run the script below to evaluate the validation images in `DMS_v1/images`. Results are stored in `evaluation_results.json`.

```
python evaluation.py --jit_path DMS46_v1.pt --data_path DMS_v1
```

## Inference on Photos

Step 1. Copy images to a single directory.

Step 2. Run the script below.

```
python inference.py --jit_path DMS46_v1.pt --image_folder path/to/images --output_folder path/to/results
```

## Code License

See [LICENSE](LICENSE.txt).
