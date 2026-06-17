<h1 align="center">Volume Transformer: Revisiting Vanilla Transformers for 3D Scene Understanding</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.19609">Paper</a>
  ·
  <a href="http://vision.rwth-aachen.de/Volt">Project Page</a>
  ·
  <a href="#citation">BibTeX</a>
</p>

<p align="center">
  This repository contains the official implementation of Volume Transformer (Volt).
</p>

<p align="center">
  <img src="https://omnomnom.vision.rwth-aachen.de/data/Volt/Volt.jpg" alt="main_figure" width="900" />
</p>

<p align="center">
  Volt partitions the input 3D scene into non-overlapping volumetric patches and embeds each patch into a token with a linear tokenizer. The resulting token sequence is processed by a Transformer encoder with global attention. The latent tokens are then upsampled back to the voxel resolution with a single transposed convolution and mapped to semantic predictions by a linear classification head.
</p>

<p align="center">
  The core Volt model implementation can be found in
  <a href="pointcept/models/volt/volt_base.py"><code>pointcept/models/volt/volt_base.py</code></a>.
</p>

## 📢 News

- 2026-06-13: Volt is now implemented in pure PyTorch and no longer depends on spconv.
- 2026-06-03: Volt won 3 challenges at CVPR 2026.
- 2026-04-22: Code release.

## Setup

This repository is built on top of [Pointcept](https://github.com/Pointcept/Pointcept/blob/04a0232b70f5c7091ffdc6bfe7a476e3eb7daff2) and incorporates components from [SGIFormer](https://github.com/RayYoh/SGIFormer/blob/4c05d57bbbd676b6a2398b03deac916e603a9dd7) for instance segmentation. For integrating image features with 3D backbones, please refer to our [DITR](https://github.com/VisualComputingInstitute/ditr) codebase.

### Dependencies
We recommend using [`uv`](https://docs.astral.sh/uv/#highlights), a fast Python package and environment manager, to install the environment.

To install `uv` on macOS and Linux, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the environment with:
```bash
# Make sure to load CUDA 12.6 beforehand
# This will automatically create a virtual environment (.venv) and install dependencies from pyproject.toml
uv sync
source .venv/bin/activate
```

## Data Preprocessing
Follow the dataset setup instructions in the [Pointcept README](https://github.com/Pointcept/Pointcept/blob/04a0232b70f5c7091ffdc6bfe7a476e3eb7daff2/README.md).

### Indoor Datasets
Preprocessing for indoor datasets is identical to Pointcept.

### SceneFun3D Dataset
Run the preprocessing script below.
```bash
python pointcept/datasets/preprocessing/scenefun3d/preprocess_scenefun3d.py --dataset_root ${SCENEFUN3D_DIR} --output_root ${PROCESSED_SCENEFUN3D_DIR}
```

### Nuscenes
For **nuScenes**, run the preprocessing script below. Unlike Pointcept preprocessing, we additionally write panoptic labels to the `.pkl` files.
```bash
uv run --no-project --python 3.12 --with nuscenes-devkit python pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py --dataset_root ${NUSCENES_DIR} --output_root ${PROCESSED_NUSCENES_DIR}
```

### SemanticKITTI
For **SemanticKITTI**, run the following script to generate the instance database used for instance CutMix.

```bash
python pointcept/datasets/preprocessing/semantic_kitti/build_instance_db_h5.py --dataset_root ${KITTI_DIR} --output_root "data/semantic_kitti_instances"
```

### Waymo
For **Waymo**, run the preprocessing script below. Waymo provides multiple LiDAR sensors. Unlike Pointcept preprocessing, we use only the points from the TOP LiDAR sensor, since only those points have semantic labels.

```bash
uv run --no-project --python 3.10 --with waymo-open-dataset-tf-2-11-0 python pointcept/datasets/preprocessing/waymo/preprocess_waymo.py --dataset_root ${WAYMO_DIR} --output_root ${PROCESSED_WAYMO_DIR} --splits training validation --num_workers ${NUM_WORKERS}
```

## Train

Download UNet teacher weights from [HuggingFace](https://huggingface.co/KadirYilmaz/Volt/tree/main)

```bash
hf download KadirYilmaz/Volt --include "teacher_weights/*.pth" --local-dir weights/
```
Then, run the training script with the `semseg-volt-distill` config for each dataset.

```bash
### ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-volt-distill -n semseg-volt-distill
### ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-volt-distill -n semseg-volt-distill
### ScanNet++
sh scripts/train.sh -g 4 -d scannetpp -c semseg-volt-distill -n semseg-volt-distill
### NuScenes
sh scripts/train.sh -g 4 -d nuscenes -c semseg-volt-distill -n semseg-volt-distill
### SemanticKITTI
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-volt-distill -n semseg-volt-distill
### Waymo
sh scripts/train.sh -g 4 -d waymo -c semseg-volt-distill -n semseg-volt-distill
```

For joint training, use the `semseg-volt-joint-small` config instead.
```bash
### ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-volt-joint-small -n semseg-volt-joint-small
### ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-volt-joint-small -n semseg-volt-joint-small
### NuScenes
sh scripts/train.sh -g 4 -d nuscenes -c semseg-volt-joint-small -n semseg-volt-joint-small
### SemanticKITTI
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-volt-joint-small -n semseg-volt-joint-small
### Waymo
sh scripts/train.sh -g 4 -d waymo -c semseg-volt-joint-small -n semseg-volt-joint-small
```

### Instance Segmentation

First, run the preprocessing script to generate superpoints for ScanNet and ScanNet200.
```bash
python pointcept/datasets/preprocessing/scannet/preprocess_superpoints.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

Download the pretrained Volt backbone weights from [HuggingFace](https://huggingface.co/KadirYilmaz/Volt/tree/main)
```bash
mkdir -p weights
curl -L -o weights/volt-small-scannet.pth https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_small/scannet/model/model_last.pth
curl -L -o weights/volt-base-scannet.pth https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_base/scannet/model/model_last.pth
curl -L -o weights/volt-small-scannet200.pth https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_small/scannet200/model/model_last.pth
curl -L -o weights/volt-base-scannet200.pth https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_base/scannet200/model/model_last.pth
```
Alternatively you can train them yourself using the corresponding configs above.

Then, run the training script with the `insseg-spformer-volt-S-0-base` config for scannet/scannet200

```bash
### ScanNet
sh scripts/train.sh -g 4 -d scannet -c insseg-spformer-volt-S-0-base -n insseg-volt
### ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c insseg-spformer-volt-S-0-base -n insseg-volt
```

For SceneFun3D, we train the model for semantic segmentation only, and use a simple clustering-based post-processing algorithm for instance segmentation. This is done automatically in the SceneFun3DTester and the results are saved in the correct format.
```bash
### SceneFun3D
sh scripts/train.sh -g 4 -d scenefun3d -c semseg-volt-base -n insseg-volt
```

## Model Zoo

We provide the experiment directories, including configs, logs, and checkpoints. The experiments can also be seen from [Hugging Face](https://huggingface.co/KadirYilmaz/Volt/tree/main).

### Semantic Segmentation: Single-Dataset Training

| Model | Dataset | Val mIoU | Exp. Dir |
| :--- | :--- | :---: | :---: |
| Volt-S | ScanNet | 77.3 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/scannet) |
| Volt-S | ScanNet200 | 36.1 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/scannet200) |
| Volt-S | ScanNet++ | 50.2 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/scannetpp) |
| Volt-S | nuScenes | 81.1 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/nuscenes) |
| Volt-S | SemanticKITTI | 70.3 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/semantic_kitti) |
| Volt-S | Waymo | 71.2 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/waymo) |

### Semantic Segmentation: Joint Training

| Model | Dataset | Val mIoU | Exp. Dir |
| :--- | :--- | :---: | :---: |
| Volt-S | ScanNet | 80.2 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_small/scannet) |
| Volt-B | ScanNet | 80.5 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_base/scannet) |
| Volt-S | ScanNet200 | 38.5 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_small/scannet200) |
| Volt-B | ScanNet200 | 40.0 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_base/scannet200) |
| Volt-S | nuScenes | 81.8 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_small/nuscenes) |
| Volt-B | nuScenes | 82.2 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_base/nuscenes) |
| Volt-S | SemanticKITTI | 72.8 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_small/semantic_kitti) |
| Volt-S | Waymo | 72.5 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/joint_training_small/waymo) |

## Citation

If you use our work in your research, please use the following BibTeX entry.

```
@article{yilmaz2026volt,
  title     = {{Volume Transformer: Revisiting Vanilla Transformers for 3D Scene Understanding}},
  author    = {Yilmaz, Kadir and Kruse, Adrian and Höfer, Tristan and de Geus, Daan and Leibe, Bastian},
  journal   = {arXiv preprint arXiv:2604.19609},
  year      = {2026}
}
```
