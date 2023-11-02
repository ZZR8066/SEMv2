import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class Neck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')):
        super(Neck, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            act_cfg=None)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        out = self.reduction_conv(out)
        return (out, )


def build_neck(cfg):
    neck = Neck(
        cfg['in_channels'],
        cfg['out_channels']
    )
    return neck