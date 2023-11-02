import os
import torch


device = torch.device('cuda')


# train dataset
train_lrcs_path = [
    '/work1/cv1/jszhang6/TSR/dataprocess/process_tal/output/train/table_v2.lrc'
]
train_max_pixel_nums = 800 * 800 * 10
train_bucket_seps = (64, 64, 64)
train_max_batch_size = 6
train_num_workers = 0


# valid dataset
valid_lrc_path = '/work1/cv1/jszhang6/TSR/dataprocess/process_tal/output/valid/table_v2.lrc'
valid_num_workers = 0
valid_batch_size = 1


# model params
# backbone
backbone=dict(
    type='HRNet',
    extra=dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(32, 64)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(32, 64, 128)),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(32, 64, 128, 256))),
    init_cfg=dict(type='Pretrained', checkpoint="/yrfs2/cv1/jszhang6/zrzhang6/PretrainModel/HRNet/hrnetv2p_w32_pretrain_on_coco.pth")
)

# neck
neck=dict(
    in_channels=[32, 64, 128, 256],
    out_channels=256
)


# posemb
posemb=dict(
    in_channels=256
)

# resa
use_resa = False

# bbox_head
bbox_head=dict(
    type='CenterNetHead',
    num_classes=1, # row/col start centern bbox detection
    in_channel=256,
    feat_channel=64,
    loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
    loss_wh=dict(type='L1Loss', loss_weight=0.1),
    loss_offset=dict(type='L1Loss', loss_weight=1.0)
)

# row split head
row_split_head=dict(
    split_type='1d', # '1d' '2d' --> whether to use CE and CTC loss
    line_type='row',
    down_stride=8,
    resa=dict(
        iters=3,
        line_type='row',
        channel=256,
        spatial_kernel=5
    ),
    in_channels=256,
    loss=dict(
        type='focal', # 'bce', 'focal', 'dice'
        div='pos', # 'pos' 'all' --> divided by positive samples or all samples,
        factor=1.
    )
)

# col split head
col_split_head=dict(
    split_type='1d', # '1d' '2d' --> whether to use CE and CTC loss
    line_type='col',
    down_stride=8,
    resa=dict(
        iters=3,
        line_type='col',
        channel=256,
        spatial_kernel=5
    ),
    in_channels=256,
    loss=dict(
        type='focal', # 'bce', 'focal', 'dice'
        div='pos', # 'pos' 'all' --> divided by positive samples or all samples,
        factor=1.
    )
)

# grid extractor
grid_extractor=dict(
    in_channels=256,
    out_channels=512,
    grid_type='bbox', # 'bbox', 'obb'
    pool_size=(3,3),
    scale=0.25,
    num_attention_layers=1,
    num_attention_heads=8,
    intermediate_size=1024, # unused parameter
    dropout_prob=0.1 # unused parameter
)

# merge head
merge_head=dict(
    in_channels=512,
    num_kernel_layers=3,
    loss=dict(
        type='focal', # 'bce', 'focal', 'dice'
        div='pos', # 'pos' 'all' --> divided by positive samples or all samples,
        factor=1.
    )
)


# train params
base_lr = 1e-4
min_lr = 1e-6
weight_decay = 0

num_epochs = 121
start_eval = 10
eval_epochs = 3
sync_rate = 20

log_sep = 20

work_dir = '../experiments/default'
train_checkpoint = None
eval_checkpoint = os.path.join(work_dir, 'best_f1_model.pth')