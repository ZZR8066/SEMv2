import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class RESA(nn.Module):

    def __init__(self, iters, line_type, channel, spatial_kernel):
        super(RESA, self).__init__()
        assert line_type in ['row', 'col']
        self.line_type = line_type
        self.iters = iters

        if line_type == 'row':
            kernel_size = (spatial_kernel, 1)
            padding_size = (spatial_kernel // 2, 0)
            # left to right and right to left
            for i in range(iters):
                conv_vert1 = ConvModule(channel, channel, kernel_size=kernel_size, stride=1, padding=padding_size, act_cfg=dict(type='ReLU'))
                conv_vert2 = ConvModule(channel, channel, kernel_size=kernel_size, stride=1, padding=padding_size, act_cfg=dict(type='ReLU'))
                setattr(self, 'conv_r2l' + str(i), conv_vert1)
                setattr(self, 'conv_l2r' + str(i), conv_vert2)
        else:
            kernel_size = (1, spatial_kernel)
            padding_size = (0, spatial_kernel // 2)
            # top to bottom
            for i in range(iters):
                conv_hori1 = ConvModule(channel, channel, kernel_size=kernel_size, stride=1, padding=padding_size, act_cfg=dict(type='ReLU'))
                conv_hori2 = ConvModule(channel, channel, kernel_size=kernel_size, stride=1, padding=padding_size, act_cfg=dict(type='ReLU'))
                setattr(self, 'conv_t2b' + str(i), conv_hori1)
                setattr(self, 'conv_b2t' + str(i), conv_hori2)

    def forward(self, feats):
        _, _, H, W = feats.shape
        if self.line_type == 'row':
            delta_lst = [W // 2 ** (self.iters - i) for i in range(self.iters)]
            # left to right
            for i in range(self.iters):
                conv = getattr(self, 'conv_' + 'l2r' + str(i))
                idx = (torch.arange(W) + delta_lst[i]) % W
                feats = feats + conv(feats[:, :, :, idx])
            # right to left
            for i in range(self.iters):
                conv = getattr(self, 'conv_' + 'r2l' + str(i))
                idx = (torch.arange(W) - delta_lst[i]) % W
                feats = feats + conv(feats[:, :, :, idx])
        else:
            delta_lst = [H // 2 ** (self.iters - i) for i in range(self.iters)]
            # top to bottom
            for i in range(self.iters):
                conv = getattr(self, 'conv_' + 't2b' + str(i))
                idx = (torch.arange(H) + delta_lst[i]) % H
                feats = feats + conv(feats[:, :, idx, :])
            # bottom to top
            for i in range(self.iters):
                conv = getattr(self, 'conv_' + 'b2t' + str(i))
                idx = (torch.arange(H) - delta_lst[i]) % H
                feats = feats + conv(feats[:, :, idx, :])
        
        return feats


def build_resa_head(cfg):
    resa = RESA(
        cfg['iters'],
        cfg['line_type'],
        cfg['channel'],
        cfg['spatial_kernel']
    )
    return resa