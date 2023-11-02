import math
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from collections import OrderedDict
from .resa_head import build_resa_head


def gen_masks(feats, images_shape, stride):
    '''
    feats: shape as [B, C, H, W]
    images_shape: shape as [B, 2] --> [[image_w, image_h],[image_w, image_h]]
    stride: the down sample stride for images_shape
    '''
    device = feats.device
    batch_size, _, H, W = feats.shape
    masks = torch.zeros([batch_size, H, W], dtype=torch.float, device=device)
    for batch_idx in range(batch_size):
        masks[batch_idx, :int(images_shape[batch_idx][1]/stride), :int(images_shape[batch_idx][0]/stride)] = 1.
    return masks


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


def compute_locations(shape, device, line_type):
    if line_type == 'row':
        pos = torch.arange(0, shape[-2], step=1, \
            dtype=torch.float32, device=device)
        pos = pos.reshape((1, 1, -1, 1))
        pos = pos.repeat(shape[0], shape[1], 1, shape[3])
    else: # 'col'
        pos = torch.arange(0, shape[-1], step=1, \
            dtype=torch.float32, device=device)
        pos = pos.reshape((1, 1, 1, -1))
        pos = pos.repeat(shape[0], shape[1], shape[2], 1)
    return pos


class SplitHead(nn.Module):
    def __init__(self, line_type, down_stride, resa, in_channels, loss):
        super().__init__()
        assert line_type in ['row', 'col']
        self.line_type = line_type

        # init down sample block --- Kernel banch
        blocks = OrderedDict()
        down_kernel_size = (1, 2) if line_type =='row' else (2,1)
        for i in range(int(math.log2(down_stride))):
            name_prefix = 'downsample' + str(i + 1)
            blocks[name_prefix + '_maxpool'] = nn.MaxPool2d(down_kernel_size)
            blocks[name_prefix + '_conv'] = ConvModule(in_channels, in_channels, \
                kernel_size=3, stride=1, padding=1, act_cfg=dict(type='ReLU'))
        self.down_sample_blocks = nn.Sequential(blocks)
        self.resa_head = build_resa_head(resa)
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

    def forward(self, feats, pad_shape, image_shape, det_bboxes, segm_labels):
        kernels = self.kernel_conv(self.resa_head(self.down_sample_blocks(feats))) # (B, C, H, W/S) or (B, C, H/S, W)
        feats = self.feats_conv(feats) # (B, C, H, W)
        _, C, _, _ = feats.shape
        stride = pad_shape[-1] / feats.shape[-1]
        masks = gen_masks(feats, image_shape, stride) # (B, H, W)

        h, w = feats.shape[-2:]
        if self.line_type == 'row':
            segments_logits = []
            center_points = []
            kernels = kernels.mean(-1) # (B, C+1, H)
            for batch_idx, bboxes in enumerate(det_bboxes):
                _, y1, _, y2 = torch.split(bboxes/stride, 1, dim=-1)
                yc = (y1 + y2) / 2 # (N, 1)
                assert yc.min().long() >= 0 and yc.max().long() <= h-1, print('Numerical Overflow')
                center_points.append(yc*stride)

                valid_kernels = kernels[batch_idx, :, yc.long().squeeze(-1)].permute(1,0).contiguous() # (N, C+1)
                weights, bias = parse_dynamic_params(valid_kernels, C, 1) # (N, C, 1, 1), (N, 1, 1, 1)
                # repeat feats (1, C, H, W) to (1, C*N, H, W)
                feats_pb = feats[batch_idx:batch_idx+1,].repeat(1, valid_kernels.size(0), 1, 1)
                segments_logit = F.conv2d(feats_pb, weight=weights, bias=bias, stride=1, padding=0, groups=valid_kernels.size(0)) # (1,N,H,W)
                segments_logit = segments_logit.squeeze(0) # (N, H, W)
                segments_logit = segments_logit.masked_fill((1-masks[batch_idx])[None,:,:].to(torch.bool), float(-1e8)) # remove padding pixels
                segments_logits.append(segments_logit)
        else:
            segments_logits = []
            center_points = []
            kernels = kernels.mean(-2) # (B, C+1, W)
            for batch_idx, bboxes in enumerate(det_bboxes):
                x1, _, x2, _ = torch.split(bboxes/stride, 1, dim=-1)
                xc = (x1 + x2) / 2 # (N, 1)
                assert xc.min().long() >= 0 and xc.max().long() <= w-1, print('Numerical Overflow')
                center_points.append(xc*stride)

                valid_kernels = kernels[batch_idx, :, xc.long().squeeze(-1)].permute(1,0).contiguous() # (N, C+1)
                weights, bias = parse_dynamic_params(valid_kernels, C, 1) # (N, C, 1, 1), (N, 1, 1, 1)
                # repeat feats (1, C, H, W) to (1, C*N, H, W)
                feats_pb = feats[batch_idx:batch_idx+1,].repeat(1, valid_kernels.size(0), 1, 1)
                segments_logit = F.conv2d(feats_pb, weight=weights, bias=bias, stride=1, padding=0, groups=valid_kernels.size(0)) # (1,N,H,W)
                segments_logit = segments_logit.squeeze(0) # (N, H, W)
                segments_logit = segments_logit.masked_fill((1-masks[batch_idx])[None,:,:].to(torch.bool), float(-1e8)) # remove padding pixels
                segments_logits.append(segments_logit)

        result_info = dict()
        if self.training:
            batch_loss = []
            for batch_idx in range(feats.size(0)):
                target_labels = F.interpolate(segm_labels[batch_idx].unsqueeze(1).float(), \
                    size=segments_logits[batch_idx].shape[-2:], mode='nearest').squeeze(1)
                if self.loss_type == 'bce':
                    segments_loss = F.binary_cross_entropy_with_logits(
                        segments_logits[batch_idx], # (N, H, W)
                        target_labels[:segments_logits[batch_idx].size(0)], # (N, H, W)
                        reduction='none'
                    )
                elif self.loss_type == 'focal':
                    segments_loss = sigmoid_focal_loss(
                        segments_logits[batch_idx], # (N, H, W)
                        target_labels[:segments_logits[batch_idx].size(0)], # (N, H, W)
                        reduction='none'
                    )
                else: # dice loss
                    segments_loss = dice_loss(
                        segments_logits[batch_idx], # (N, H, W)
                        target_labels[:segments_logits[batch_idx].size(0)], # (N, H, W)
                        reduction='none'
                    )
                if self.loss_div == 'all':
                    N = segments_logits[batch_idx].size(0)
                    segments_loss = self.loss_factor * (segments_loss * masks[batch_idx, None]).sum() / (masks[batch_idx].sum()*N + 1e-5)
                else: # divided by positive samples
                    if target_labels.sum() == 0.:
                        segments_loss = 0 * self.loss_factor * (segments_loss * masks[batch_idx, None]).sum() / (target_labels.sum() + 1e-5)
                    else:
                        segments_loss = self.loss_factor * (segments_loss * masks[batch_idx, None]).sum() / (target_labels.sum() + 1e-5)
                batch_loss.append(segments_loss)
            result_info['split_loss'] = sum(batch_loss) / len(batch_loss)

        if self.training:
            if self.line_type == 'row':
                target_labels = F.interpolate(segm_labels.float(), size=segments_logits[0].shape[-2:], mode='nearest') # (B, NumRow, H, W)
                pos = compute_locations(target_labels.shape, target_labels.device, self.line_type) # (B, NumRow, H, W)
                target_pos = (target_labels * pos).sum(-2) / (target_labels.sum(-2) + 1e-5) # (B, NumRow, W)
                avail_masks =  target_labels.max(-2)[0] # (B, NumRow, W)
                assert avail_masks.max() == 1
                batch_loss = []
                for batch_idx in range(target_labels.size(0)):
                    avail_logits = segments_logits[batch_idx] # (AvailRow, H, W)
                    avail_logits = avail_logits.softmax(-2) # (AvailRow, H, W)
                    avail_len = avail_logits.size(0)
                    avail_pos = pos[batch_idx, :avail_len] # (AvailRow, H, W)
                    pred_pos = (avail_logits * avail_pos).sum(-2) # (AvailRow, W)

                    target_pos_pb = target_pos[batch_idx, :avail_len] # (AvailRow, W)
                    mask = avail_masks[batch_idx, :avail_len] # (AvailRow, W)
                    
                    line_loss = F.l1_loss(pred_pos * mask, target_pos_pb * mask, reduction='none')
                    line_loss = line_loss.sum() / (mask.sum() + 1e-4) / 20
                    batch_loss.append(line_loss)
            else:
                target_labels = F.interpolate(segm_labels.float(), size=segments_logits[0].shape[-2:], mode='nearest') # (B, NumCol, H, W)
                pos = compute_locations(target_labels.shape, target_labels.device, self.line_type) # (B, NumCol, H, W)
                target_pos = (target_labels * pos).sum(-1) / (target_labels.sum(-1) + 1e-5) # (B, NumCol, H)
                avail_masks =  target_labels.max(-1)[0] # (B, NumCol, H)
                assert avail_masks.max() == 1
                batch_loss = []
                for batch_idx in range(target_labels.size(0)):
                    avail_logits = segments_logits[batch_idx] # (AvailCol, H, W)
                    avail_logits = avail_logits.softmax(-1) # (AvailCol, H, W)
                    avail_len = avail_logits.size(0)
                    avail_pos = pos[batch_idx, :avail_len] # (AvailCol, H, W)
                    pred_pos = (avail_logits * avail_pos).sum(-1) # (AvailCol, H)

                    target_pos_pb = target_pos[batch_idx, :avail_len] # (AvailCol, H)
                    mask = avail_masks[batch_idx, :avail_len] # (AvailCol, H)
                    
                    line_loss = F.l1_loss(pred_pos * mask, target_pos_pb * mask, reduction='none')
                    line_loss = line_loss.sum() / (mask.sum() + 1e-4) / 20
                    batch_loss.append(line_loss)
            result_info['split_line_loss'] = sum(batch_loss) / len(batch_loss)

        return result_info, segments_logits, center_points


def build_split_head(cfg):
    split_head = SplitHead(
        cfg['line_type'],
        cfg['down_stride'],
        cfg['resa'],
        cfg['in_channels'],
        cfg['loss']
    )
    return split_head