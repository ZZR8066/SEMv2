from .split_head_1d import build_split_head as build_split_head_1d
from .split_head_2d import build_split_head as build_split_head_2d


def build_split_head(cfg):
    if cfg['split_type'] == '1d':
        return build_split_head_1d(cfg)
    else:
        return build_split_head_2d(cfg)