"""
Build SemanticKITTI instance database (HDF5).
"""

import argparse
import os
from tqdm import tqdm
import numpy as np
import h5py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pointcept.datasets.utils import coords_to_normals

LEARNING_MAP = {
    0: -1,  # "unlabeled"
    1: -1,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: -1,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 8,  # "lane-marking" to "road" ---------------------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: -1,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,  # "moving-car" to "car" ------------------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,  # "moving-person" to "person" ------------------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,  # "moving-truck" to "truck" --------------------------------mapped
    259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

THINGS_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
TRAIN_SEQS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
VAL_SEQS = [8]


def process_scan(data_path):
    results = []
    with open(data_path, "rb") as b:
        scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
    coord = scan[:, :3]
    strength = scan[:, -1].reshape([-1, 1])
    normals = coords_to_normals(coord)

    label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
    with open(label_file, "rb") as a:
        panoptic = np.fromfile(a, dtype=np.int32).reshape(-1)
        instance = (panoptic >> 16).astype(np.int32) - 1
        segment = np.vectorize(LEARNING_MAP.__getitem__)(panoptic & 0xFFFF).astype(
            np.int32
        )

    valid = np.isin(segment, THINGS_CLASSES) & (instance >= 0)
    for p_id in np.unique(panoptic[valid]):
        mask = panoptic == p_id

        inst_coord = coord[mask]
        inst_strength = strength[mask]
        inst_normals = normals[mask]
        inst_sem_class = segment[mask][0]

        # Stacking (N,3) coords + (N,1) strength + (N,3) normals = (N, 7)
        combined_data = np.concatenate(
            [inst_coord, inst_strength, inst_normals], axis=1
        )
        results.append((inst_sem_class, data_path, p_id, combined_data))
    return results


def write_to_h5(all_results, output_path):
    print(f"Writing to {output_path}...")
    with h5py.File(output_path, "w") as f:
        counts = {cls: 0 for cls in THINGS_CLASSES}
        for inst_sem_class, data_path, p_id, combined_data in all_results:
            group_name = f"class_{inst_sem_class}"
            if group_name not in f:
                f.create_group(group_name)
            ds = f[group_name].create_dataset(
                str(counts[inst_sem_class]),
                data=combined_data,
                compression="gzip",
                compression_opts=4,
            )
            ds.attrs["data_path"] = data_path
            ds.attrs["panoptic_label"] = p_id
            counts[inst_sem_class] += 1

        # Store the final counts as attributes
        for cls, count in counts.items():
            if count > 0:
                f[f"class_{cls}"].attrs["count"] = count
                print(f"Class {cls}: {count} instances")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_workers", default=mp.cpu_count(), type=int)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    data_list = []
    for seq in TRAIN_SEQS:
        seq = str(seq).zfill(2)
        seq_folder = os.path.join(args.dataset_root, "dataset", "sequences", seq)
        seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        data_list += [os.path.join(seq_folder, "velodyne", file) for file in seq_files]

    all_results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_scan, dp) for dp in data_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            all_results.extend(future.result())

    output_path = os.path.join(args.output_root, "train.h5")
    write_to_h5(all_results, output_path)

    data_list = []
    for seq in VAL_SEQS:
        seq = str(seq).zfill(2)
        seq_folder = os.path.join(args.dataset_root, "dataset", "sequences", seq)
        seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        data_list += [os.path.join(seq_folder, "velodyne", file) for file in seq_files]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_scan, dp) for dp in data_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            all_results.extend(future.result())

    output_path = os.path.join(args.output_root, "trainval.h5")
    write_to_h5(all_results, output_path)


if __name__ == "__main__":
    main()
