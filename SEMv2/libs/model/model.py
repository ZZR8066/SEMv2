import torch
from torch import nn
from mmdet.models import build_backbone, build_head
from .neck import build_neck
from .posemb import build_posemb_head
from .resa_head import build_resa_head
from .split_head import build_split_head
from .grid_extractor import build_grid_extractor
from .merge_head import build_merge_head
from .utils import cal_bbox_head_loss, parse_segm_label, parse_grid_bboxes, decode_bbox_head_results, cal_bbox_f1, refine_gt_bboxes


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck(cfg.neck)
        self.posemb = build_posemb_head(cfg.posemb)

        self.row_split_head = build_split_head(cfg.row_split_head)
        self.col_split_head = build_split_head(cfg.col_split_head)
        self.grid_extractor = build_grid_extractor(cfg.grid_extractor)
        self.merge_head = build_merge_head(cfg.merge_head)
    
    def forward(self, images, images_size, row_start_bboxes=None, row_line_masks=None, \
        col_start_bboxes=None, col_line_masks=None, layouts=None):
        result_info = dict()

        feats = self.neck(self.backbone(images)) # (B, C, H/4, W/4)
        feats = tuple([self.posemb(feats[0])]) # (B, C, H/4, W/4)
        
        # row table line instance segmentation
        rs_result_info, row_segm_logits, row_center_points = self.row_split_head(feats[0], images.shape, images_size, row_start_bboxes, row_line_masks)
        rs_result_info = {'row_%s' % key: val for key, val in rs_result_info.items()}
        result_info.update(rs_result_info)

        # col table line instance segmentation
        cs_result_info, col_segm_logits, col_center_points = self.col_split_head(feats[0], images.shape, images_size, col_start_bboxes, col_line_masks)
        cs_result_info = {'col_%s' % key: val for key, val in cs_result_info.items()}
        result_info.update(cs_result_info)

        # parse the table grid bboxes
        stride_w = images.shape[3] / feats[0].shape[3]
        stride_h = images.shape[2] / feats[0].shape[2]
        if self.training:
            table_gird_bboxes, num_rows, num_cols = parse_grid_bboxes(row_center_points, parse_segm_label(row_segm_logits, row_line_masks), \
                col_center_points, parse_segm_label(col_segm_logits, col_line_masks), stride_w, stride_h) # batch tensor -> [(N, 4)]
        else:
            table_gird_bboxes, num_rows, num_cols = parse_grid_bboxes(row_center_points, row_segm_logits, \
                col_center_points, col_segm_logits, stride_w, stride_h, score_threshold=0.25) # batch tensor -> [(N, 4)]
        
        # extract the grid-level features
        grid_feats, grid_masks = self.grid_extractor(feats[0], num_rows, num_cols, table_gird_bboxes, images_size)

        # grid merge predict
        mg_result_info, mg_logits = self.merge_head(grid_feats, grid_masks, layouts)
        mg_result_info = {'merge_%s' % key: val for key, val in mg_result_info.items()}
        result_info.update(mg_result_info)
        
        return result_info, row_start_bboxes, row_center_points, row_segm_logits, \
            col_start_bboxes, col_center_points, col_segm_logits, mg_logits, num_rows, num_cols