"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        if "img_num" in batch[0].keys():
            max_img_num = max([d["img_num"] for d in batch])
        batch = {
            key: (
                (
                    collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        collate_fn(
                            [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                        ),
                        dim=0,
                    )
                )
                if "correspondence" not in key
                else collate_fn(
                    [
                        F.pad(
                            d[key].permute(0, 2, 1),
                            (0, max_img_num - d[key].shape[1]),
                            value=-1,
                        ).permute(0, 2, 1)
                        for d in batch
                    ]
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    unmixed_offset = batch["offset"].clone() if "offset" in batch else None
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            instance_ignore_index = -1
            vx_offset = batch.get("offset", None)
            pt_offset = batch.get("origin_offset", None)
            if vx_offset is None:
                return batch
            for i in range(0, len(vx_offset) - 1, 2):
                vx0s = 0 if i == 0 else int(vx_offset[i - 1].item())
                vx0e = int(vx_offset[i].item())
                vx1s = int(vx_offset[i + 1 - 1].item())
                vx1e = int(vx_offset[i + 1].item())
                vx_shift = vx0e - vx0s

                inst0 = batch["instance"][vx0s:vx0e]
                valid0 = inst0 != instance_ignore_index
                inst_shift = int(inst0[valid0].max().item()) + 1 if valid0.any() else 0
                if inst_shift:
                    inst1 = batch["instance"][vx1s:vx1e]
                    valid1 = inst1 != instance_ignore_index
                    inst1[valid1] += inst_shift

                if pt_offset is not None:
                    pt0s = 0 if i == 0 else int(pt_offset[i - 1].item())
                    pt0e = int(pt_offset[i].item())
                    pt1s = int(pt_offset[i + 1 - 1].item())
                    pt1e = int(pt_offset[i + 1].item())

                    if "origin_instance" in batch:
                        oinst0 = batch["origin_instance"][pt0s:pt0e]
                        valid0 = oinst0 != instance_ignore_index
                        oinst_shift = (
                            int(oinst0[valid0].max().item()) + 1 if valid0.any() else 0
                        )
                        if oinst_shift:
                            oinst1 = batch["origin_instance"][pt1s:pt1e]
                            valid1 = oinst1 != instance_ignore_index
                            oinst1[valid1] += oinst_shift

                    if "inverse" in batch and vx_shift:
                        batch["inverse"][pt1s:pt1e] += vx_shift

                    if "superpoint" in batch:
                        sp0 = batch["superpoint"][pt0s:pt0e]
                        valid0 = sp0 >= 0
                        sp_shift = (
                            int(sp0[valid0].max().item()) + 1 if valid0.any() else 0
                        )
                        if sp_shift:
                            sp1 = batch["superpoint"][pt1s:pt1e]
                            valid1 = sp1 >= 0
                            sp1[valid1] += sp_shift

            if vx_offset.numel() > 1:
                batch["offset"] = torch.cat(
                    [vx_offset[1:-1:2], vx_offset[-1].unsqueeze(0)], dim=0
                )
            if pt_offset is not None:
                batch["unmixed_origin_offset"] = pt_offset.clone()
                if pt_offset.numel() > 1:
                    batch["origin_offset"] = torch.cat(
                        [pt_offset[1:-1:2], pt_offset[-1].unsqueeze(0)], dim=0
                    )
            offset_assets = [
                asset
                for asset in batch.keys()
                if "offset" in asset
                and asset
                not in (
                    "offset",
                    "origin_offset",
                    "unmixed_offset",
                    "unmixed_origin_offset",
                )
            ]

        else:
            offset_assets = [asset for asset in batch.keys() if "offset" in asset]
        for offset_asset in offset_assets:
            batch[offset_asset] = torch.cat(
                [batch[offset_asset][1:-1:2], batch[offset_asset][-1].unsqueeze(0)],
                dim=0,
            )
        if "img_num" in batch.keys():
            n = batch["img_num"].shape[0]
            num_pairs = n // 2
            len_pairs = num_pairs * 2
            pairs_tensor = batch["img_num"][:len_pairs]

            if num_pairs == 0:
                pass
            else:
                summed_pairs = pairs_tensor.view(-1, 2).sum(dim=1)
                if n % 2 != 0:
                    last_element = batch["img_num"][-1:]
                    result = torch.cat((summed_pairs, last_element))
                else:
                    result = summed_pairs
                batch["img_num"] = result
        correspondence_assets = [
            asset for asset in batch.keys() if "correspondence" in asset
        ]
        for correspondence_asset in correspondence_assets:
            offset = batch["offset"]
            start = 0
            N, v, n = batch[correspondence_asset].shape
            v2 = v * 2
            batch_correspondence_mix = -torch.ones((N, v2, n))
            for i in range(len(offset)):
                if i % 2 == 0:
                    batch_correspondence_mix[start : offset[i], 0:v] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                if i % 2 != 0:
                    batch_correspondence_mix[start : offset[i], v:] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                start = offset[i]
            if len(offset) % 2 == 0:
                pass
            else:
                start = 0 if len(offset) == 1 else offset[-2]
                batch_correspondence_mix[start:N, -v:] = batch[correspondence_asset][
                    start:N, -v:
                ]
            batch[correspondence_asset] = batch_correspondence_mix
    if unmixed_offset is not None:
        batch["unmixed_offset"] = unmixed_offset
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
