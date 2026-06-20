import os

import numpy as np
import torch

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset

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


@DATASETS.register_module()
class ARKitScenesLabelMakerDataset(DefaultDataset):
    label_key = "semantic_pseudo_gt_wn199"

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


@DATASETS.register_module()
class ARKitScenesLabelMakerScanNet200Dataset(ARKitScenesLabelMakerDataset):
    label_key = "semantic_pseudo_gt_scannet200"
