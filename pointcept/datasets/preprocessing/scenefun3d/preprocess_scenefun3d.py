import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

CLASSES = {
    "rotate": 0,
    "key_press": 1,
    "tip_push": 2,
    "hook_pull": 3,
    "pinch_pull": 4,
    "hook_turn": 5,
    "foot_push": 6,
    "plug_in": 7,
    "unplug": 8,
}

IGNORE_INDEX = -1


def estimate_normals(coords, radius=0.1, max_nn=100):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    return np.asarray(pcd.normals).astype(np.float32)


def read_list(p):
    p = Path(p)
    if not p.exists():
        return set()
    return {l.strip() for l in p.read_text().splitlines() if l.strip()}


def get_split(scene_id, train_scenes, val_scenes, test_scenes):
    if scene_id in train_scenes:
        return "train"
    if scene_id in val_scenes:
        return "val"
    if scene_id in test_scenes:
        return "test"
    raise ValueError(f"Scene {scene_id} not found in any split")


def process_scene(ply_path, output_root, train_scenes, val_scenes, test_scenes):
    ply_path = Path(ply_path)
    scene_id = ply_path.parent.name

    split = get_split(scene_id, train_scenes, val_scenes, test_scenes)

    ply = PlyData.read(ply_path)
    vertices = ply["vertex"].data

    coords = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float32
    )

    colors = np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8)

    num_points = len(coords)

    semantic_gt = np.full(num_points, IGNORE_INDEX, dtype=np.int16)
    instance_gt = np.full(num_points, IGNORE_INDEX, dtype=np.int16)

    if split != "test":
        annotation_path = ply_path.with_name(f"{scene_id}_annotations.json")

        with open(annotation_path) as f:
            annotations = json.load(f)["annotations"]

        instance_id = 1
        for annotation in annotations:
            label = annotation["label"]

            if label == "exclude":
                continue

            class_id = CLASSES[label]
            indices = np.asarray(annotation["indices"], dtype=np.int64)

            semantic_gt[indices] = class_id
            instance_gt[indices] = instance_id

            instance_id += 1

    crop_mask_path = ply_path.with_name(f"{scene_id}_crop_mask.npy")
    crop_mask = np.load(crop_mask_path)

    coords = coords[crop_mask]
    colors = colors[crop_mask]
    # Shift semantic labels by 1 to use background class as 0
    semantic_gt = semantic_gt[crop_mask] + 1
    instance_gt = instance_gt[crop_mask]

    normals = estimate_normals(coords)

    scene_output_dir = output_root / split / scene_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(scene_output_dir / "coord.npy", coords)
    np.save(scene_output_dir / "color.npy", colors)
    np.save(scene_output_dir / "normal.npy", normals)
    np.save(scene_output_dir / "segment.npy", semantic_gt)
    np.save(scene_output_dir / "instance.npy", instance_gt)
    np.save(scene_output_dir / "crop_mask.npy", crop_mask)

    return scene_id, split, len(coords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the SceneFun3D dataset",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()

    dataset_root = Path(config.dataset_root)
    output_root = Path(config.output_root)

    meta_root = Path(os.path.dirname(__file__)) / "meta_data"
    train_scenes = read_list(meta_root / "train_scenes.txt")
    val_scenes = read_list(meta_root / "val_scenes.txt")
    test_scenes = read_list(meta_root / "test_scenes.txt")

    ply_paths = sorted(dataset_root.glob("*/*_laser_scan.ply"))
    print(
        f"Found {len(ply_paths)} scenes. Processing with {config.num_workers} workers..."
    )

    with ThreadPoolExecutor(max_workers=config.num_workers) as pool:
        for scene_id, split, num_points in pool.map(
            process_scene,
            ply_paths,
            repeat(output_root),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(test_scenes),
        ):
            print(f"Processed {split}/{scene_id}: {num_points} points")


if __name__ == "__main__":
    main()
