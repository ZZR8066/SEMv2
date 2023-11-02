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


def cal_segments(cls_logits):
    cls_probs, cls_ids = torch.max(cls_logits, dim=1)
    cls_probs = cls_probs.tolist()
    cls_ids = cls_ids.tolist()

    segments = list()

    for idx in range(len(cls_ids)):
        if idx == 0:
            if cls_ids[idx] == 1:
                segments.append([idx, cls_probs[idx]])
        elif cls_ids[idx] == 1:
            if cls_ids[idx - 1] == 1:
                if cls_probs[idx] > segments[-1][1]:
                    segments[-1] = [idx, cls_probs[idx]]
            else:
                segments.append([idx, cls_probs[idx]])
    
    segments = [item[0] for item in segments]
    if len(segments) < 2:
        segments = [0, cls_logits.shape[0]-1] # é¦–å°¾
    return segments


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


def cal_segment_loss(cls_logits, det_bboxes, stride, line_type):
    device = cls_logits.device
    bg_cls_loss = list()
    fg_ctc_loss = list()
    for cls_logits_pi, bboxes in zip(cls_logits, det_bboxes):
        if line_type == 'row':
            _, start_ids, _, end_ids = torch.split(bboxes/stride, 1, dim=-1) # (N, 1)
        else:
            start_ids, _, end_ids, _ = torch.split(bboxes/stride, 1, dim=-1) # (N, 1)
        bg_target = torch.full([cls_logits_pi.shape[0]], 0, dtype=torch.long, device=device)
        for start_id, end_id in zip(start_ids, end_ids):
            bg_target[int(start_id):int(end_id+1)] = -1 # remove foreground pixel
        bg_cls_loss_pi = F.cross_entropy(cls_logits_pi, bg_target, ignore_index=-1)
        bg_cls_loss.append(bg_cls_loss_pi)

        if start_ids.size(0) > 0:
            fg_logits = [cls_logits_pi[int(start_id):int(end_id+1), :] for start_id, end_id in zip(start_ids, end_ids)]
            fg_logits_length = [item.shape[0] for item in fg_logits]
            fg_max_length = max(fg_logits_length)

            fg_logits = torch.stack([F.pad(item, (0, 0, 0, fg_max_length-item.shape[0])) for item in fg_logits], dim=1)
            fg_logits = torch.log_softmax(fg_logits, dim=2)
            fg_logits_length = torch.tensor(fg_logits_length, dtype=torch.long, device=device)

            fg_target = torch.ones([fg_logits_length.shape[0], 1], dtype=torch.long, device=device)
            fg_target_length = torch.ones_like(fg_logits_length)

            fg_ctc_loss_pi = F.ctc_loss(fg_logits, fg_target, fg_logits_length, fg_target_length, zero_infinity=True)
            fg_ctc_loss.append(fg_ctc_loss_pi)

    bg_cls_loss = torch.mean(torch.stack(bg_cls_loss))
    fg_ctc_loss = torch.mean(torch.stack(fg_ctc_loss))
    return bg_cls_loss, fg_ctc_loss


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


class SplitHead(nn.Module):
    def __init__(self, line_type, down_stride, resa, in_channels, loss):
        super().__init__()
        assert line_type in ['row', 'col']
        self.line_type = line_type

        # init down sample block --- Detection banch
        blocks = OrderedDict()
        down_kernel_size = (1, 2) if line_type =='row' else (2,1)
        for i in range(int(math.log2(down_stride))):
            name_prefix = 'downsample' + str(i + 1)
            blocks[name_prefix + '_maxpool'] = nn.MaxPool2d(down_kernel_size)
            blocks[name_prefix + '_conv'] = ConvModule(in_channels, in_channels, \
                kernel_size=3, stride=1, padding=1, act_cfg=dict(type='ReLU'))
        self.det_down_sample_blocks = nn.Sequential(blocks)
        self.det_resa_head = build_resa_head(resa)
        self.det_conv = ConvModule(
            in_channels,
            2, # background/foreground prob
            kernel_size=1,
            conv_cfg=None,
            act_cfg=None)

        # init down sample block --- Kernel banch
        blocks = OrderedDict()
        down_kernel_size = (1, 2) if line_type =='row' else (2,1)
        for i in range(int(math.log2(down_stride))):
            name_prefix = 'downsample' + str(i + 1)
            blocks[name_prefix + '_maxpool'] = nn.MaxPool2d(down_kernel_size)
            blocks[name_prefix + '_conv'] = ConvModule(in_channels, in_channels, \
                kernel_size=3, stride=1, padding=1, act_cfg=dict(type='ReLU'))
        self.down_sample_blocks = nn.Sequential(blocks)
        self.kernel_resa_head = build_resa_head(resa)
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
        det_feats = self.det_conv(self.det_resa_head(self.det_down_sample_blocks(feats))) # (B, 2, H, W/S) or (B, 2, H/S, W)
        kernels = self.kernel_conv(self.kernel_resa_head(self.down_sample_blocks(feats))) # (B, C, H, W/S) or (B, C, H/S, W)
        feats = self.feats_conv(feats) # (B, C, H, W)
        _, C, _, _ = feats.shape
        stride = pad_shape[-1] / feats.shape[-1]
        masks = gen_masks(feats, image_shape, stride) # (B, H, W)

        h, w = feats.shape[-2:]
        if self.line_type == 'row':
            segments_logits = []
            center_points = []
            det_feats = det_feats.mean(-1).transpose(1, 2).contiguous() # (B, H, 2)
            kernels = kernels.mean(-1) # (B, C+1, H)
            for batch_idx in range(det_feats.shape[0]):
                # parsing the kernel position
                det_feats_pb = det_feats[batch_idx][:int(image_shape[batch_idx][1]/stride+2)] # (valid_H, 2)
                if not self.training: # inference stage
                    yc = torch.tensor(cal_segments(det_feats_pb), device=kernels.device).unsqueeze(1) # (N, 1)
                else: # during training stage, select from max prob from foreground
                    _, y1, _, y2 = torch.split(det_bboxes[batch_idx]/stride, 1, dim=-1) # (N, 1)
                    yc = (y1 + y2) / 2 # (N, 1)
                    cls_probs_pi = torch.softmax(det_feats_pb, dim=1)[:, 1]
                    for row_idx in range(yc.size(0)):
                        height = y2[row_idx] - y1[row_idx]
                        if height > 1 and y2[row_idx]+1 < det_feats_pb.shape[0]:
                            span_cls_probs = cls_probs_pi[int(y1[row_idx][0]):int(y2[row_idx][0]+1)]
                            yc[row_idx][0] = torch.argmax(span_cls_probs).item() + y1[row_idx][0]
                assert yc.min().long() >= 0 and yc.max().long() <= h-1, print('Numerical Overflow')
                center_points.append(yc*stride)

                # extract kernels
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
            det_feats = det_feats.mean(-2).transpose(1, 2).contiguous() # (B, W, 2)
            kernels = kernels.mean(-2) # (B, C+1, W)
            for batch_idx in range(det_feats.shape[0]):
                # parsing the kernel position 
                det_feats_pb = det_feats[batch_idx][:int(image_shape[batch_idx][0]/stride+2)] # (valid_W, 2)
                if not self.training: # inference stage
                    xc = torch.tensor(cal_segments(det_feats_pb), device=kernels.device).unsqueeze(1) # (N, 1)
                else:
                    x1, _, x2, _ = torch.split(det_bboxes[batch_idx]/stride, 1, dim=-1) # (N, 1)
                    xc = (x1 + x2) / 2 # (N, 1)
                    cls_probs_pi = torch.softmax(det_feats_pb, dim=1)[:, 1]
                    for col_idx in range(xc.size(0)):
                        width = x2[col_idx] - x1[col_idx]
                        if width > 1 and x2[col_idx] + 1 < det_feats_pb.shape[0]:
                            span_cls_probs = cls_probs_pi[int(x1[col_idx][0]):int(x2[col_idx][0]+1)]
                            xc[col_idx][0] = torch.argmax(span_cls_probs).item() + x1[col_idx][0]
                assert xc.min().long() >= 0 and xc.max().long() <= w-1, print('Numerical Overflow')
                center_points.append(xc*stride)

                # extract kernels
                valid_kernels = kernels[batch_idx, :, xc.long().squeeze(-1)].permute(1,0).contiguous() # (N, C+1)
                weights, bias = parse_dynamic_params(valid_kernels, C, 1) # (N, C, 1, 1), (N, 1, 1, 1)
                # repeat feats (1, C, H, W) to (1, C*N, H, W)
                feats_pb = feats[batch_idx:batch_idx+1,].repeat(1, valid_kernels.size(0), 1, 1)
                segments_logit = F.conv2d(feats_pb, weight=weights, bias=bias, stride=1, padding=0, groups=valid_kernels.size(0)) # (1,N,H,W)
                segments_logit = segments_logit.squeeze(0) # (N, H, W)
                segments_logit = segments_logit.masked_fill((1-masks[batch_idx])[None,:,:].to(torch.bool), float(-1e8)) # remove padding pixels
                segments_logits.append(segments_logit)

        result_info = dict()
        # BCE and CTC loss for segment
        if self.training:
            bg_cls_loss, fg_ctc_loss = cal_segment_loss(det_feats, det_bboxes, stride, self.line_type)
            result_info['bg_cls_loss'] = bg_cls_loss
            result_info['fg_ctc_loss'] = fg_ctc_loss

        # segment loss
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
            result_info['split_loss'] = sum(batch_loss)/len(batch_loss)

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