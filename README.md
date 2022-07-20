# The Dense Material Segmentation Dataset

The Dense Material Segmentation Dataset (DMS) consists of 3 million polygon labels of material categories (metal, wood, glass, etc) for 44 thousand RGB images. The dataset is described in the research paper, [A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing](https://arxiv.org).

## Downloading

Our data consists of annotations in the form of label maps. The corresponding RGB images are part of [Open Images](https://storage.googleapis.com/openimages/web/index.html) and must be acquired separately.

This archive contains the fused label maps which are the majority label across multiple annotators. [Download](https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms_v1_labels.zip) (630 MB). This is the only archive needed to train and evaluate material segmentation models.

This archive contains a pre-trained model which predicts 46 kinds of materials. [Download](https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms46_v1.zip) (170 MB).

This archive contains the polygon annotations which were used to create the fused labels. [Download](https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms_v1_polygons.zip) (2.3 GB).

## Data License

The dataset is licensed for non-commercial use under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The pre-trained model is licensed under [LICENSE](LICENSE.txt).

Licenses for the original RGB images must be acquired separately. See [Open Images](https://storage.googleapis.com/openimages/web/index.html) for further information.
