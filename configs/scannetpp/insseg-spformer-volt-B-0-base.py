_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

epoch = 200
eval_epoch = 100

weight = "weights/volt-base-scannetpp.pth"

batch_size = 8
num_worker = 48
mix_prob = 0.8
empty_cache = False
enable_amp = True
use_ema = False
clip_grad = 10.0
evaluate = True
find_unused_parameters = True

# ScanNet++ keeps 100 raw semantic ids, but only 84 are instance-valid classes.
num_classes = 100
segment_ignore_index = (-1, 0, 1, 2, 16, 19, 20, 24, 26, 33, 36, 48, 53, 63, 64, 73, 74)
semantic_num_classes = 84
num_channels = 256

enable_wandb = True

model = dict(
    type="SPFormer-v1m1",
    backbone=dict(
        type="Volt",
        in_channels=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        up_mlp_dim=256,
    ),
    decoder=dict(
        type="SPFormerDecoder",
        num_class=semantic_num_classes,
        in_channel=num_channels,
        num_layer=6,
        num_query=800,
        d_model=384,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="gelu",
        iter_pred=True,
        attn_mask=True,
        use_query_pos=False,
        use_score=False,
        use_param_query=True,
    ),
    criterion=dict(
        type="SPFormerCriterion",
        matcher=dict(
            type="SPFormerHungarianMatcher",
            costs=[
                dict(type="SPFormerQueryClassificationCost", weight=0.5),
                dict(type="SPFormerMaskBCECost", weight=1.0),
                dict(type="SPFormerMaskDiceCost", weight=1.0),
            ],
        ),
        loss_weight=[0.2, 1.0, 1.0, 0.5],
        num_classes=semantic_num_classes,
        non_object_weight=0.1,
        fix_dice_loss_weight=False,
        iter_matcher=True,
        fix_mean_loss=True,
    ),
    semantic_num_classes=semantic_num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    topk_insts=400,
    score_thr=0.0,
    npoint_thr=100,
    nms=True,
)

optimizer = dict(type="AdamW", lr=0.0003, weight_decay=0.1)
scheduler = dict(type="PolyLR")

dataset_type = "ScanNetPPDataset"
data_root = "data/scannetpp"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="SphereCrop", point_max=1000000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            ),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.95
            ),
            dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis="x", p=0.95),
            dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis="y", p=0.95),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            dict(type="RandomShift", shift=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.5, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "instance": "origin_instance",
                    "segment": "origin_segment",
                },
            ),
            dict(
                type="Update",
                keys_dict={
                    "index_valid_keys": [
                        "coord",
                        "color",
                        "normal",
                        "strength",
                        "segment",
                        "instance",
                        "image_coord",
                        "image_mask",
                    ]
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "origin_coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "instance",
                    "origin_instance",
                    "superpoint",
                    "inverse",
                ),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="MeanShift"),
            dict(
                type="RandomRotate",
                angle=[0.35, 0.35],
                axis="z",
                center=[0, 0, 0],
                always_apply=True,
            ),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "instance": "origin_instance",
                    "segment": "origin_segment",
                },
            ),
            dict(
                type="Update",
                keys_dict={
                    "index_valid_keys": [
                        "coord",
                        "color",
                        "normal",
                        "strength",
                        "segment",
                        "instance",
                        "image_coord",
                        "image_mask",
                    ]
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "origin_coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "instance",
                    "origin_instance",
                    "superpoint",
                    "inverse",
                    "name",
                ),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
)
data["test"] = data["val"]

hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=True,
)
