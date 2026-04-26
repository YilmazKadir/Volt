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

## Model Zoo

We provide the experiment directories, including configs, logs, and checkpoints. The experiments can also be seen from [Hugging Face](https://huggingface.co/KadirYilmaz/Volt/tree/main).

### Semantic Segmentation: Single-Dataset Training

| Model | Dataset | Val mIoU | Exp. Dir |
| :--- | :--- | :---: | :---: |
| Volt-S | ScanNet | 76.3 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/scannet) |
| Volt-S | ScanNet200 | 36.1 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/scannet200) |
| Volt-S | ScanNet++ | 50.2 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/scannetpp) |
| Volt-S | nuScenes | 81.1 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/nuscenes) |
| Volt-S | SemanticKITTI | 70.3 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/semantic_kitti) |
| Volt-S | Waymo | 71.2 | [link](https://huggingface.co/KadirYilmaz/Volt/tree/main/Volt_experiments/single_dataset/waymo) |

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
