_base_ = ["../custom_import.py"]

dataset_type = 'PST900Dataset'
data_root = '/home/ulindu/datasets/PST900_RGBT_Dataset'
img_scale = (1280, 720)
crop_size = (128, 128)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='ToRGB'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type='RandomCrop', crop_size=(448, 448)),
            dict(type="RandomFlip"),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(type="Collect", keys=["img"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'flip', 'img_info']),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='ToRGB'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(type="Collect", keys=["img"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'flip', 'img_info']),
        ],
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/rgb",
        ann_dir="train/labels",
        pipeline=train_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="test/rgb",
        ann_dir="test/labels",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
