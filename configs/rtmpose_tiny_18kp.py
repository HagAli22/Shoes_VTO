"""
rtmpose_tiny_18kp.py
────────────────────
MMPose config for RTMPose-Tiny with 18 custom foot keypoints (9/foot).

This config extends the official RTMPose-tiny body config to:
  1. Replace the 17-KP COCO head with an 18-KP foot head
  2. Add custom foot keypoint metadata (names, sigmas, skeleton)
  3. Use our FootKeypointDataset for training/validation

Usage (MMPose CLI)
------------------
  mim train mmpose configs/rtmpose_tiny_18kp.py \\
      --work-dir outputs/rtmpose_tiny_foot \\
      [--resume outputs/rtmpose_tiny_foot/last.pth]

Reference
---------
  https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose
"""

import os

# ── Runtime ──────────────────────────────────────────────────────────────────
default_scope      = "mmpose"
default_hooks      = dict(
    timer          = dict(type="IterTimerHook"),
    logger         = dict(type="LoggerHook", interval=50),
    param_scheduler= dict(type="ParamSchedulerHook"),
    checkpoint     = dict(type="CheckpointHook", interval=10, max_keep_ckpts=3,
                          save_best="coco/AP", rule="greater"),
    sampler_seed   = dict(type="DistSamplerSeedHook"),
    visualization  = dict(type="PoseVisualizationHook", enable=False),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level      = "INFO"
load_from      = None
resume         = False

# ── Dataset metadata ─────────────────────────────────────────────────────────
dataset_info = dict(
    dataset_name = "foot_keypoint_dataset",
    paper_info   = dict(
        title   = "Shoe AR Try-On AI Perception Layer",
        authors = "Internal",
        year    = 2026,
    ),
    keypoint_info = {
        0:  dict(name="left_big_toe",         id=0,  color=[255,128,0],  type="lower", swap="right_big_toe"),
        1:  dict(name="left_little_toe",      id=1,  color=[255,153,51], type="lower", swap="right_little_toe"),
        2:  dict(name="left_near_little_toe", id=2,  color=[255,178,102],type="lower", swap="right_near_little_toe"),
        3:  dict(name="left_near_big_toe",    id=3,  color=[255,204,153],type="lower", swap="right_near_big_toe"),
        4:  dict(name="left_far_big_toe",     id=4,  color=[255,229,204],type="lower", swap="right_far_big_toe"),
        5:  dict(name="left_far_little_toe",  id=5,  color=[102,178,255],type="lower", swap="right_far_little_toe"),
        6:  dict(name="left_dorsum",          id=6,  color=[51,153,255], type="lower", swap="right_dorsum"),
        7:  dict(name="left_heel",            id=7,  color=[0,128,255],  type="lower", swap="right_heel"),
        8:  dict(name="left_upper_heel",      id=8,  color=[0,102,204],  type="lower", swap="right_upper_heel"),
        9:  dict(name="right_big_toe",        id=9,  color=[255,0,0],    type="lower", swap="left_big_toe"),
        10: dict(name="right_little_toe",     id=10, color=[255,51,51],  type="lower", swap="left_little_toe"),
        11: dict(name="right_near_little_toe",id=11, color=[255,102,102],type="lower", swap="left_near_little_toe"),
        12: dict(name="right_near_big_toe",   id=12, color=[255,153,153],type="lower", swap="left_near_big_toe"),
        13: dict(name="right_far_big_toe",    id=13, color=[255,204,204],type="lower", swap="left_far_big_toe"),
        14: dict(name="right_far_little_toe", id=14, color=[204,255,204],type="lower", swap="left_far_little_toe"),
        15: dict(name="right_dorsum",         id=15, color=[153,255,153],type="lower", swap="left_dorsum"),
        16: dict(name="right_heel",           id=16, color=[102,255,102],type="lower", swap="left_heel"),
        17: dict(name="right_upper_heel",     id=17, color=[51,255,51],  type="lower", swap="left_upper_heel"),
    },
    skeleton_info = {
        # Left foot
        0: dict(link=("left_big_toe",        "left_little_toe"),        id=0,  color=[255,128,0]),
        1: dict(link=("left_little_toe",      "left_near_little_toe"),   id=1,  color=[255,153,51]),
        2: dict(link=("left_near_little_toe", "left_far_little_toe"),    id=2,  color=[255,178,102]),
        3: dict(link=("left_far_little_toe",  "left_far_big_toe"),       id=3,  color=[255,204,153]),
        4: dict(link=("left_far_big_toe",     "left_near_big_toe"),      id=4,  color=[255,229,204]),
        5: dict(link=("left_near_big_toe",    "left_big_toe"),           id=5,  color=[102,178,255]),
        6: dict(link=("left_big_toe",         "left_heel"),              id=6,  color=[51,153,255]),
        7: dict(link=("left_heel",            "left_upper_heel"),        id=7,  color=[0,128,255]),
        8: dict(link=("left_dorsum",          "left_heel"),              id=8,  color=[0,102,204]),
        # Right foot
        9:  dict(link=("right_big_toe",        "right_little_toe"),       id=9,  color=[255,0,0]),
        10: dict(link=("right_little_toe",     "right_near_little_toe"),  id=10, color=[255,51,51]),
        11: dict(link=("right_near_little_toe","right_far_little_toe"),   id=11, color=[255,102,102]),
        12: dict(link=("right_far_little_toe", "right_far_big_toe"),      id=12, color=[255,153,153]),
        13: dict(link=("right_far_big_toe",    "right_near_big_toe"),     id=13, color=[255,204,204]),
        14: dict(link=("right_near_big_toe",   "right_big_toe"),          id=14, color=[204,255,204]),
        15: dict(link=("right_big_toe",        "right_heel"),             id=15, color=[153,255,153]),
        16: dict(link=("right_heel",           "right_upper_heel"),       id=16, color=[102,255,102]),
        17: dict(link=("right_dorsum",         "right_heel"),             id=17, color=[51,255,51]),
    },
    joint_weights = [
        1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 1.0, 1.2, 1.2,   # left foot
        1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 1.0, 1.2, 1.2,   # right foot
    ],
    sigmas = [
        0.025, 0.025, 0.035, 0.035, 0.040, 0.040, 0.040, 0.030, 0.035,
        0.025, 0.025, 0.035, 0.035, 0.040, 0.040, 0.040, 0.030, 0.035,
    ],
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            # Download: https://download.openmmlab.com/mmpose/v1/cspnext-pretrained/
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                       "cspnext-tiny_udep-aic-coco_210e-256x192-cbed682d_20230130.pth",
        ),
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=(96,),
        out_channels=None,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=384,
        out_channels=18,          # ← 18 foot keypoints
        input_size=(256, 192),
        in_featuremap_size=(8, 6),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss",
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=dict(
            type="SimCCLabel",
            input_size=(256, 192),
            smoothing_type="gaussian",
            sigma=(4.9, 5.66),
            normalize=False,
            use_dark=False,
        ),
    ),
    test_cfg=dict(flip_test=True),
)

# ── Optimizer ─────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# ── LR Schedule ───────────────────────────────────────────────────────────────
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.0,
        begin=150,
        end=200,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
auto_scale_lr = dict(base_batch_size=1024)
train_cfg     = dict(by_epoch=True, max_epochs=200, val_interval=5)
val_cfg       = dict()
test_cfg      = dict()

# ── Data Pipeline ─────────────────────────────────────────────────────────────
codec = dict(
    type="SimCCLabel",
    input_size=(256, 192),
    smoothing_type="gaussian",
    sigma=(4.9, 5.66),
    normalize=False,
    use_dark=False,
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(type="CoarseDropout", max_holes=1, max_height=0.4,
                 max_width=0.4, min_holes=1, min_height=0.2, min_width=0.2, p=0.5),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# ── Datasets ──────────────────────────────────────────────────────────────────
_DATA_ROOT    = "data/processed/"
_TRAIN_JSON   = "data/annotations/train_annotations.json"
_VAL_JSON     = "data/annotations/val_annotations.json"

dataset_type  = "CocoDataset"
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=_DATA_ROOT,
        data_mode="topdown",
        ann_file=_TRAIN_JSON,
        pipeline=train_pipeline,
        metainfo=dataset_info,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=_DATA_ROOT,
        data_mode="topdown",
        ann_file=_VAL_JSON,
        pipeline=val_pipeline,
        test_mode=True,
        metainfo=dataset_info,
    ),
)
test_dataloader = val_dataloader

# ── Evaluators ────────────────────────────────────────────────────────────────
val_evaluator = dict(
    type="CocoMetric",
    ann_file=_VAL_JSON,
    nms_mode="none",
    score_mode="bbox",
)
test_evaluator = val_evaluator

