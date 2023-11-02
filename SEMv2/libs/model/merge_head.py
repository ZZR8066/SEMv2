import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .utils import cal_merge_acc


def parse_dynamic_params(params,
                         weight_nums,
                         bias_nums,
                         out_channels=1):
    assert params.dim() == 2
    # params: (num_ins, n_param)

    num_insts = params.size(0)
    weight_splits, bias_splits = torch.split(params, [weight_nums, bias_nums], dim=1)

    # (out_channels, in_channels, 1, 1)
    weight_splits = weight_splits.reshape(num_insts * out_channels, -1, 1, 1)
    bias_splits = bias_splits.reshape(num_insts * out_channels)

    return weight_splits, bias_splits


def parse_segm_labels(layouts):
    b, nr, nc = layouts.shape
    segm_labels = torch.zeros((b, nr*nc, nr ,nc), dtype=torch.float, device=layouts.device)
    for batch_idx in range(b):
        for row_idx in range(nr):
            for col_idx in range(nc):
                segm_labels[batch_idx, row_idx*nc+col_idx] = (layouts[batch_idx] == layouts[batch_idx, row_idx, col_idx]).float()
    return segm_labels


def sigmoid_focal_loss( pred,
                        target,
                        weight=1.0,
                        gamma=2.0,
                        alpha=0.25,
                        reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def dice_loss(pred, target, reduction='mean'):
    pred = pred.sigmoid()
    target = target.type_as(pred)

    a = pred * target # |X?Y|
    b = pred * pred + 1e-3  # |X|
    c = target * target + 1e-3  # |Y|
    d = (2 * a) / (b + c)
    loss = 1 - d
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


class MergeHead(nn.Module):
    def __init__(self, in_channels, num_kernel_layers, loss):
        super().__init__()
        self.pre_kernel_convs = nn.Sequential(*nn.ModuleList([ConvModule(in_channels, in_channels, \
            kernel_size=3, stride=1, padding=1, act_cfg=dict(type='ReLU')) for _ in range(num_kernel_layers)]))

        # init kernel conv -- Kernel banch
        self.kernel_conv = ConvModule(
            in_channels,
            in_channels+1, # in_channels + bias
            kernel_size=1,
            conv_cfg=None,
            act_cfg=None)

        # init feature conv -- Feature banch
        self.feats_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            conv_cfg=None,
            act_cfg=None)
        
        assert loss['type'] in ['bce', 'focal', 'dice']
        self.loss_type = loss['type']
        assert loss['div'] in ['pos', 'all']
        self.loss_div = loss['div']
        self.loss_factor = loss['factor']

    def forward(self, feats, masks, layouts):
        kernels = self.kernel_conv(self.pre_kernel_convs(feats)) # (B, C+1, H, W)
        feats = self.feats_conv(feats) # (B, C, H, W)
        B, C, _, _ = feats.shape

        kernels = kernels.permute(0, 2, 3, 1).contiguous().reshape(B, -1, C+1)
        segments_logits = []
        for batch_idx in range(B):
            valid_kernels = kernels[batch_idx]
            weights, bias = parse_dynamic_params(valid_kernels, C, 1) # (N, C, 1, 1), (N, 1, 1, 1)
            # repeat feats (1, C, H, W) to (1, C*N, H, W)
            feats_pb = feats[batch_idx:batch_idx+1,].repeat(1, valid_kernels.size(0), 1, 1)
            segments_logit = F.conv2d(feats_pb, weight=weights, bias=bias, stride=1, padding=0, groups=valid_kernels.size(0)) # (1,N,H,W)
            segments_logit = segments_logit.squeeze(0) # (N, H, W)
            segments_logit = segments_logit.masked_fill((1-masks[batch_idx])[None,:,:].to(torch.bool), float(-1e8)) # remove padding pixels
            segments_logits.append(segments_logit)
        segments_logits = torch.stack(segments_logits, dim=0)

        result_info = dict()
        if self.training:
            batch_loss = []
            segm_labels = parse_segm_labels(layouts)
            for batch_idx in range(segments_logits.size(0)):
                # cal merge loss
                if self.loss_type == 'bce':
                    segments_loss = F.binary_cross_entropy_with_logits(
                        segments_logits[batch_idx], # (N, H, W)
                        segm_labels.float()[batch_idx], # (N, H, W)
                        reduction='none',
                    )
                elif self.loss_type == 'focal':
                    segments_loss = sigmoid_focal_loss(
                        segments_logits[batch_idx], # (N, H, W)
                        segm_labels.float()[batch_idx], # (N, H, W)
                        reduction='none',
                    )
                else: # dice loss
                    segments_loss = dice_loss(
                        segments_logits[batch_idx], # (N, H, W)
                        segm_labels.float()[batch_idx], # (N, H, W)
                        reduction='none',
                    )
                grid_num = int(masks[batch_idx].sum())
                if self.loss_div == 'all':
                    segments_loss = self.loss_factor * ((segments_loss * masks[batch_idx, None]).sum(-1).sum(-1) * masks[batch_idx].reshape(-1)).sum() / (grid_num*grid_num + 1e-5)
                else:
                    segments_loss = self.loss_factor * ((segments_loss * masks[batch_idx, None]).sum(-1).sum(-1) * masks[batch_idx].reshape(-1)).sum() / (grid_num + 1e-5)
                batch_loss.append(segments_loss)
            # cal merge acc
            acc = cal_merge_acc((segments_logits.data.sigmoid() > 0.5).float(), segm_labels, masks)
            result_info = dict(loss=sum(batch_loss)/len(batch_loss), acc=acc)

        return result_info, segments_logits


def build_merge_head(cfg):
    merge_head = MergeHead(
        cfg['in_channels'],
        cfg['num_kernel_layers'],
        cfg['loss']
    )
    return merge_head