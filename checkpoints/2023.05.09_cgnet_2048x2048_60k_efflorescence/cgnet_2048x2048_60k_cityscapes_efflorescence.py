norm_cfg = dict(type='SyncBN', eps=0.001, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CGNet',
        norm_cfg=dict(type='SyncBN', eps=0.001, requires_grad=True),
        in_channels=3,
        num_channels=(32, 64, 128),
        num_blocks=(3, 21),
        dilations=(2, 4),
        reductions=(8, 16)),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', eps=0.001, requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[2.5959933, 10.055647])),
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
dataset_type = 'CityscapesDatasetEfflorescence'
data_root = '/home/user/Downloads/raw'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (2048, 2048)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='DilateAnnotations'),
    dict(type='Resize', img_scale=(2048, 4096), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(2048, 2048), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(2048, 2048), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 4096),
        flip=False,
        transforms=[
            dict(type='DilateAnnotations'),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CityscapesDatasetEfflorescence',
        data_root='/home/user/Downloads/raw',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            # dict(type='DilateAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 4096), ratio_range=(0.5, 2.0)),
            dict(
                type='RandomCrop', crop_size=(2048, 2048), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(2048, 2048), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CityscapesDatasetEfflorescence',
        data_root='/home/user/Downloads/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 4096),
                flip=False,
                transforms=[
                    # dict(type='DilateAnnotations'),
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDatasetEfflorescence',
        data_root='/home/user/Downloads/raw',
        img_dir='leftImg8bit/test',
        ann_dir='gtFine/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 4096),
                flip=False,
                transforms=[
                    # dict(type='DilateAnnotations'),
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
total_iters = 60000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
work_dir = './work_dirs/cgnet_2048x2048_60k_cityscapes_efflorescence'
gpu_ids = range(0, 2)
auto_resume = False
