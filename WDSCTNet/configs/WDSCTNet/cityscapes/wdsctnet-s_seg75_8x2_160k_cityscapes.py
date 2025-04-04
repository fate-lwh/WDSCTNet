#1. model config
checkpoint_teacher = 'pretrain/Teacher_SegFormer_B3_city.pth'
checkpoint_backbone = 'pretrain/WDSCT-S_Pretrain.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_Distill',
    pretrained=None,
    backbone=dict(
        type='SCTNet',
        init_cfg=dict(
            type='Pretrained',
            checkpoint= checkpoint_backbone
        ),
        base_channels=32,
        spp_channels=64),
    decode_head=dict(
        type='SCTHead',
        in_channels=128,
        channels=128,
        dropout_ratio=0.0,
        in_index=0,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='RMILoss',
            num_classes=19,
            rmi_radius=3,
            rmi_pool_way=0,
            rmi_pool_size=3,
            rmi_pool_stride=3,
            loss_weight_lambda=0.5,
            lambda_way=1)),
    auxiliary_head=[
        dict(
            type='DetailGuidanceHead',
            in_channels=64,
            channels=64,
            num_classes=19,
            loss_decode=dict(
                type='DetailAggregateLoss', weight=0.5, loss_weight=0.4)),
        dict(
            type='VitGuidanceHead',
            init_cfg=dict(
                type='Pretrained',
                checkpoint=checkpoint_teacher),
            in_channels=128,
            channels=128,
            base_channels=32,
            in_index=2,
            num_classes=19,
            loss_decode=dict(
                type='AlignmentLoss', loss_weight=[3, 15, 15, 15]))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

#2. data config
dataset_type = 'CityscapesDataset'
data_root = 'data/hybrid/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.25, 1.5)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1536, 768),
        flip=False,
        transforms=[
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CityscapesDataset',
        data_root='data/hybrid/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 1024),
                ratio_range=(0.25, 1.5)),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CityscapesDataset',
        data_root='data/hybrid/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1536, 768),
                flip=False,
                transforms=[
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
        type='CityscapesDataset',
        data_root='data/hybrid/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1536, 768),
                flip=False,
                transforms=[
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

#3. optimizer config
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.0125,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            teacher_backbone=dict(lr_mult=0.0),
            teacher_head=dict(lr_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=1e-06,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
find_unused_parameters = True
auto_resume = True
