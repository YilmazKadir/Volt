import os
import argparse
import glob
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
import pointseg
import open3d as o3d
import torch

CLOUD_FILE_PFIX = "_vh_clean_2"


def handle_process(scene_path, output_path, train_scenes, val_scenes):
    scene_id = os.path.basename(scene_path)
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    if scene_id in train_scenes:
        output_path = os.path.join(output_path, "train", f"{scene_id}")
        split_name = "train"
    elif scene_id in val_scenes:
        output_path = os.path.join(output_path, "val", f"{scene_id}")
        split_name = "val"
    else:
        output_path = os.path.join(output_path, "test", f"{scene_id}")
        split_name = "test"

    print(f"Processing: {scene_id} in {split_name}")
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices_sp = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces_sp = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = pointseg.segment_mesh(vertices_sp, faces_sp).numpy()
    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "superpoint.npy"), superpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()
    meta_root = Path(os.path.dirname(__file__)) / "meta_data"

    # Load train/val splits
    with open(meta_root / "scannetv2_train.txt") as train_file:
        train_scenes = train_file.read().splitlines()
    with open(meta_root / "scannetv2_val.txt") as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/scans*/scene*"))

    # Preprocess data.
    print("Processing scenes...")

    with ThreadPoolExecutor(max_workers=config.num_workers) as pool:
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(train_scenes),
            repeat(val_scenes),
        )
