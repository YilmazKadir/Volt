_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16  # bs: total bs in all gpus
num_worker = 24
mix_prob = 0.2
empty_cache = False
enable_amp = True
use_ema = True
find_unused_parameters = True

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=128,
    backbone=dict(
        type="Volt",
        in_channels=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        up_mlp_dim=128,
    ),
    teacher=dict(
        type="DefaultSegmentorV2",
        backbone=dict(
            type="SpUNet-v1m3",
            in_channels=4,
            num_classes=0,
            channels=(32, 64, 128, 256, 256, 128, 96, 96),
            layers=(2, 3, 4, 6, 2, 2, 2, 2),
            enc_mode=False,
            conditions=("nuScenes", "SemanticKITTI", "Waymo"),
            zero_init=False,
            norm_decouple=True,
            norm_adaptive=False,
            norm_affine=True,
        ),
        backbone_out_channels=96,
        conditions=("nuScenes", "SemanticKITTI", "Waymo"),
        num_classes=(16, 19, 22),
    ),
    teacher_weights="weights/teacher_weights/joint_outdoor_unet_teacher.pth",
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            label_smoothing=0.1,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    num_classes=(16, 19, 22),
)

# scheduler settings
epoch = 25
eval_epoch = 25
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

ignore_index = -1
names = [
    "Car",
    "Truck",
    "Bus",
    # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    "Other Vehicle",
    "Motorcyclist",
    "Bicyclist",
    "Pedestrian",
    "Sign",
    "Traffic Light",
    # Lamp post, traffic sign pole etc.
    "Pole",
    # Construction cone/pole.
    "Construction Cone",
    "Bicycle",
    "Motorcycle",
    "Building",
    # Bushes, tree branches, tall grasses, flowers etc.
    "Vegetation",
    "Tree Trunk",
    # Curb on the edge of roads. This does not include road boundaries if there’s no curb.
    "Curb",
    # Surface a vehicle could drive on. This includes the driveway connecting
    # parking lot and road over a section of sidewalk.
    "Road",
    # Marking on the road that’s specifically for defining lanes such as
    # single/double white/yellow lines.
    "Lane Marker",
    # Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    "Other Ground",
    # Most horizontal surface that’s not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    "Walkable",
    # Nicely paved walkable surface when pedestrians most likely to walk on.
    "Sidewalk",
]

data = dict(
    num_classes=22,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type="ConcatDataset",
        datasets=[
            # Waymo
            dict(
                type="WaymoDataset",
                split="training",
                data_root="data/waymo",
                transform=[
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(
                        type="PointClipDistance", max_dist=75.0, z_min=-4.0, z_max=2.0
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(
                        type="RandomShift",
                        shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
                    ),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Update", keys_dict={"condition": "Waymo"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=ignore_index,
                loop=1,  # sampling weight
            ),
            # SemanticKITTI
            dict(
                type="SemanticKITTIDataset",
                split="train",
                data_root="data/semantic_kitti",
                transform=[
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="InstanceCutMix",
                        db_path="data/semantic_kitti_instances/train.h5",
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(
                        type="PointClipDistance", max_dist=50.0, z_min=-4.0, z_max=2.0
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(
                        type="RandomShift",
                        shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
                    ),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="InstanceShift", p=0.5, shift_range=[4, 4, 0.5]),
                    dict(type="InstanceRotate", p=0.5, axis="z", angle=[-0.5, 0.5]),
                    dict(type="InstanceScale", p=0.5, scale=[0.9, 1.1]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Update", keys_dict={"condition": "SemanticKITTI"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=ignore_index,
                loop=1,  # sampling weight
            ),
            # nuScenes
            dict(
                type="NuScenesDataset",
                split="train",
                data_root="data/nuscenes",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[-1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(
                        type="PointClipDistance", max_dist=70.0, z_min=-4.0, z_max=2.0
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Update", keys_dict={"condition": "nuScenes"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=ignore_index,
                loop=2,  # sampling weight
            ),
        ],
    ),
    val=dict(
        type="WaymoDataset",
        split="validation",
        data_root="data/waymo",
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="PointClipDistance", max_dist=75.0, z_min=-4.0, z_max=2.0),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="Update", keys_dict={"condition": "Waymo"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "inverse",
                    "condition",
                ),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type="WaymoDataset",
        split="validation",
        data_root="data/waymo",
        transform=[
            dict(type="PointClipDistance", max_dist=75.0, z_min=-4.0, z_max=2.0),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="Update", keys_dict={"condition": "Waymo"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
