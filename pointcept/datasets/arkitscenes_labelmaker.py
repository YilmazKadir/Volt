import glob
import os
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pointcept.utils.cache import shared_dict
from pointcept.utils.logger import get_root_logger

from .builder import DATASETS
from .preprocessing.scannet.meta_data.scannet200_constants import VALID_CLASS_IDS_200
from .transform import TRANSFORMS, Compose

# Based on code from:
# https://github.com/quantaji/LabelMaker-Pointcept/blob/main/pointcept/datasets/alc.py

IGNORE_INDEX = -1
CLASS_LABELS = (
    "wall",
    "chair",
    "book",
    "cabinet",
    "door",
    "floor",
    "ashcan",
    "table",
    "window",
    "bookshelf",
    "display",
    "cushion",
    "box",
    "picture",
    "ceiling",
    "doorframe",
    "desk",
    "swivel_chair",
    "towel",
    "sofa",
    "sink",
    "backpack",
    "lamp",
    "chest_of_drawers",
    "apparel",
    "armchair",
    "bed",
    "curtain",
    "mirror",
    "plant",
    "radiator",
    "toilet_tissue",
    "shoe",
    "bag",
    "bottle",
    "countertop",
    "coffee_table",
    "toilet",
    "computer_keyboard",
    "fridge",
    "stool",
    "computer",
    "mug",
    "telephone",
    "light",
    "jacket",
    "bathtub",
    "shower_curtain",
    "microwave",
    "footstool",
    "baggage",
    "laptop",
    "printer",
    "shower_stall",
    "soap_dispenser",
    "stove",
    "fan",
    "paper",
    "stand",
    "bench",
    "wardrobe",
    "blanket",
    "booth",
    "duplicator",
    "bar",
    "soap_dish",
    "switch",
    "coffee_maker",
    "decoration",
    "range_hood",
    "blackboard",
    "clock",
    "railing",
    "mat",
    "seat",
    "bannister",
    "container",
    "mouse",
    "person",
    "stairway",
    "basket",
    "dumbbell",
    "column",
    "bucket",
    "windowsill",
    "signboard",
    "dishwasher",
    "loudspeaker",
    "washer",
    "paper_towel",
    "clothes_hamper",
    "piano",
    "sack",
    "handcart",
    "blind",
    "dish_rack",
    "mailbox",
    "bag",
    "bicycle",
    "ladder",
    "rack",
    "tray",
    "toaster",
    "paper_cutter",
    "plunger",
    "dryer",
    "guitar",
    "fire_extinguisher",
    "pitcher",
    "pipe",
    "plate",
    "vacuum",
    "bowl",
    "hat",
    "rod",
    "water_cooler",
    "kettle",
    "oven",
    "scale",
    "broom",
    "hand_blower",
    "coatrack",
    "teddy",
    "alarm_clock",
    "ironing_board",
    "fire_alarm",
    "machine",
    "music_stand",
    "fireplace",
    "furniture",
    "vase",
    "vent",
    "candle",
    "crate",
    "dustpan",
    "earphone",
    "jar",
    "projector",
    "gat",
    "step",
    "step_stool",
    "vending_machine",
    "coat",
    "coat_hanger",
    "drinking_fountain",
    "hamper",
    "thermostat",
    "banner",
    "iron",
    "soap",
    "chopping_board",
    "kitchen_island",
    "shirt",
    "sleeping_bag",
    "tire",
    "toothbrush",
    "bathrobe",
    "faucet",
    "slipper",
    "thermos",
    "tripod",
    "dispenser",
    "heater",
    "pool_table",
    "remote_control",
    "stapler",
    "treadmill",
    "beanbag",
    "dartboard",
    "metronome",
    "rope",
    "sewing_machine",
    "shredder",
    "toolbox",
    "water_heater",
    "brush",
    "control",
    "dais",
    "dollhouse",
    "envelope",
    "food",
    "frying_pan",
    "helmet",
    "tennis_racket",
    "umbrella",
)


def get_wordnet(label_key="wn199-merged-v2"):
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__)))
        / "preprocessing/arkitscenes/label_mapping.csv"
    )
    ids_found = []
    data = [{"id": 0, "name": "unknown", "color": [0, 0, 0]}]
    for row in table.index:
        if table[label_key].isnull()[row]:
            continue
        if table.loc[row, label_key] in ids_found:
            continue
        ids_found.append(table.loc[row, label_key])
        data.append(
            {
                "id": int(table.loc[row, label_key]),
                "name": table.loc[row, "wnsynsetkey"],
                "color": [int(x) for x in table.loc[row, "color"].split("-")],
            }
        )
    return data


def get_wordnet_compact_mapping():
    wordnet_info = get_wordnet()[1:]
    wordnet_info = sorted(wordnet_info, key=lambda x: x["id"])

    class2id = np.array([item["id"] for item in wordnet_info])
    id2class = np.array([IGNORE_INDEX] * (class2id.max() + 1))
    for class_, id_ in enumerate(class2id):
        id2class[id_] = class_

    return class2id, id2class


@DATASETS.register_module()
class ARKitScenesLabelMakerDataset(Dataset):
    label_key = "semantic_pseudo_gt_wn199"

    def __init__(
        self,
        split="train",
        data_root="data/arkitscenes",
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(ARKitScenesLabelMakerDataset, self).__init__()
        self.get_class_to_id()

        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()

        self.ignore_index = ignore_index

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list),
                self.loop,
                split,
            )
        )

    def get_class_to_id(self):
        self.class2id = get_wordnet_compact_mapping()[0]

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        if not self.cache:
            data = torch.load(data_path, weights_only=False)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)

        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if self.label_key in data.keys():
            segment = data[self.label_key].reshape(-1)
        else:
            segment = np.ones(coord.shape[0]) * -1
        instance = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            coord=coord,
            color=color.astype(np.float32),
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )

        if normal is not None:
            data_dict["normal"] = normal

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ARKitScenesLabelMakerScanNet200Dataset(ARKitScenesLabelMakerDataset):
    label_key = "semantic_pseudo_gt_scannet200"

    def get_class_to_id(self):
        self.class2id = np.array(VALID_CLASS_IDS_200)
