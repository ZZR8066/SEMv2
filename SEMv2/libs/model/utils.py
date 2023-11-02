import torch
import torch.nn.functional as F
from mmcv.ops import batched_nms
import numpy as np


def cal_bbox_head_loss(bbox_head, bbox_head_predicts, gt_bboxes, feat_shape, image_shape):
    target_result, avg_factor = bbox_head.get_targets(gt_bboxes, \
        [torch.zeros_like(item[:, 0], dtype=torch.long) for item in gt_bboxes], \
            feat_shape, image_shape)
        
    center_heatmap_target = target_result['center_heatmap_target']
    wh_target = target_result['wh_target']
    offset_target = target_result['offset_target']
    wh_offset_target_weight = target_result['wh_offset_target_weight']

    center_heatmap_pred, wh_pred, offset_pred = [item[0] for item in bbox_head_predicts]

    # Since the channel of wh_target and offset_target is 2, the avg_factor
    # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
    loss_center_heatmap = bbox_head.loss_center_heatmap(
        center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
    loss_wh = bbox_head.loss_wh(
        wh_pred,
        wh_target,
        wh_offset_target_weight,
        avg_factor=avg_factor * 2)
    loss_offset = bbox_head.loss_offset(
        offset_pred,
        offset_target,
        wh_offset_target_weight,
        avg_factor=avg_factor * 2)
    
    result_info = dict()
    result_info['loss_center_heatmap'] = loss_center_heatmap
    result_info['loss_wh'] = loss_wh
    result_info['loss_offset'] = loss_offset
    return result_info


def decode_bbox_head_results(bbox_head, bbox_head_predicts, pad_shape, image_shape, with_nms=False, \
    score_threshold=0.3, cfg=dict(topk=200, kernel=3, nms_cfg=dict(iou_threshold=0.7), max_per_img=200)):
    center_heatmap_pred, wh_pred, offset_pred = [item[0].data for item in bbox_head_predicts]

    # remove invalid padding pixels
    feat_h, feat_w = center_heatmap_pred.shape[-2:]
    input_h, input_w = pad_shape
    scale_h = input_h / feat_h
    scale_w = input_w / feat_w
    for batch_idx in range(center_heatmap_pred.size(0)):
        center_heatmap_pred[batch_idx, :, \
            int(image_shape[batch_idx, 1]/scale_h):, int(image_shape[batch_idx, 0]/scale_w):] = 0.
    
    # decode bboxes in input image
    batch_det_bboxes, batch_labels = bbox_head.decode_heatmap(
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        pad_shape,
        k=cfg['topk'],
        kernel=cfg['kernel']
    )
    
    # nms predict bboxes
    det_results = []
    for (det_bboxes, det_labels) in zip(batch_det_bboxes, batch_labels):
        # remove bboxes with low score
        valid_mask = det_bboxes[:, -1] > score_threshold
        det_bboxes = det_bboxes[valid_mask]
        det_labels = det_labels[valid_mask]
        if det_labels.numel() == 0:
            det_results.append(tuple([det_bboxes, det_labels]))
        elif with_nms:
            out_bboxes, keep = batched_nms(det_bboxes[:, :4].contiguous(), \
                det_bboxes[:, -1].contiguous(), det_labels, cfg['nms_cfg'])
            out_labels = det_labels[keep]

            if len(out_bboxes) > 0:
                idx = torch.argsort(out_bboxes[:, -1], descending=True)
                idx = idx[:cfg['max_per_img']]
                out_bboxes = out_bboxes[idx]
                out_labels = out_labels[idx]

            det_results.append(tuple([out_bboxes, out_labels]))
        else:
            det_results.append(tuple([det_bboxes, det_labels]))
    return det_results


def cal_bbox_f1(batch_det_bboxes, batch_gt_bboxes, iou_threshold=0.75):
    ''' calculate the F1 score of bboxes
    batch_det_bboxes: (list) [(N,4)]
    batch_gt_bboxes: (list) [(N,4)]
    iou_threshold: threshold for calculating the positive samples 
    '''
    total_tp = 0
    for det_bboxes, gt_bboxes in zip(batch_det_bboxes, batch_gt_bboxes):
        if det_bboxes.numel() == 0 or gt_bboxes.numel() == 0:
            continue

        # cal iou
        det_xmin, det_ymin, det_xmax, det_ymax = torch.split(det_bboxes, 1, dim=-1)
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = torch.split(gt_bboxes, 1, dim=-1)
        
        gt_areas = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
        det_areas = (det_xmax - det_xmin) * (det_ymax - det_ymin)

        xmin = torch.max(det_xmin[:, None], gt_xmin) # (NumDet, NumGt, 1)
        ymin = torch.max(det_ymin[:, None], gt_ymin) # (NumDet, NumGt, 1)
        xmax = torch.min(det_xmax[:, None], gt_xmax) # (NumDet, NumGt, 1)
        ymax = torch.min(det_ymax[:, None], gt_ymax) # (NumDet, NumGt, 1)

        h = (ymax - ymin).clamp_min(0) # (NumDet, NumGt, 1)
        w = (xmax - xmin).clamp_min(0) # (NumDet, NumGt, 1)
        intersect = h * w # (NumDet, NumGt, 1)

        union = det_areas[:, None] + gt_areas - intersect # (NumDet, NumGt, 1)
        iou = intersect / union # (NumDet, NumGt, 1)

        # sum true positive samples
        total_tp += (iou.max(0)[0] > iou_threshold).sum()

    precision = total_tp / (sum([item.shape[0] for item in batch_det_bboxes]) + 1e-5)
    recall = total_tp / (sum([item.shape[0] for item in batch_gt_bboxes]) + 1e-5)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    result_info = dict()
    result_info['precision'] = float(precision)
    result_info['recall'] = float(recall)
    result_info['f1'] = float(f1)
    return result_info
    

def refine_gt_bboxes(batch_gt_bboxes, batch_det_bboxes, iou_threshold=0.9):
    '''
    replace the gt_bboxes with det_bboxes according to the iou
    '''
    batch_refine_gt_bboxes = []
    for det_bboxes, gt_bboxes in zip(batch_det_bboxes, batch_gt_bboxes):
        if det_bboxes.numel() == 0 or gt_bboxes.numel() == 0:
            batch_refine_gt_bboxes.append(gt_bboxes)
            continue

        # cal iou
        det_xmin, det_ymin, det_xmax, det_ymax = torch.split(det_bboxes, 1, dim=-1)
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = torch.split(gt_bboxes, 1, dim=-1)
        
        gt_areas = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
        det_areas = (det_xmax - det_xmin) * (det_ymax - det_ymin)

        xmin = torch.max(det_xmin[:, None], gt_xmin) # (NumDet, NumGt, 1)
        ymin = torch.max(det_ymin[:, None], gt_ymin) # (NumDet, NumGt, 1)
        xmax = torch.min(det_xmax[:, None], gt_xmax) # (NumDet, NumGt, 1)
        ymax = torch.min(det_ymax[:, None], gt_ymax) # (NumDet, NumGt, 1)

        h = (ymax - ymin).clamp_min(0) # (NumDet, NumGt, 1)
        w = (xmax - xmin).clamp_min(0) # (NumDet, NumGt, 1)
        intersect = h * w # (NumDet, NumGt, 1)

        union = det_areas[:, None] + gt_areas - intersect # (NumDet, NumGt, 1)
        iou = intersect / union # (NumDet, NumGt, 1)

        # refinement gt bboxes
        values, det_indices = iou.max(0)
        gt_indices = torch.arange(det_indices.size(0), device=det_indices.device).view_as(det_indices)
        valid_det_indices = det_indices[values > iou_threshold]
        valid_gt_indices = gt_indices[values > iou_threshold]
        gt_bboxes[valid_gt_indices] = det_bboxes[valid_det_indices]

        batch_refine_gt_bboxes.append(gt_bboxes)

    return batch_refine_gt_bboxes


def parse_segm_label(segm_logits, segm_labels):
    batch_labels = []
    for batch_idx in range(len(segm_logits)):
        labels = F.interpolate(segm_labels[batch_idx].unsqueeze(1).float(), \
            size=segm_logits[batch_idx].shape[-2:], mode='nearest').squeeze(1)
        batch_labels.append(labels)
    return batch_labels


def parse_grid_bboxes(row_center_points, row_segm_logits, col_center_points, col_segm_logits,\
    stride_w, stride_h, score_threshold=0.5, kernel_size=3, radius=30):
    '''
        parse the start bboxes and segmentation results to table grid bboxes
    '''

    batch_grid_bboxes = []
    batch_num_rows = []
    batch_num_cols = []
    for batch_idx in range(len(row_center_points)):
        # parse start point
        rs_yc = row_center_points[batch_idx].long() # (NumRow, 1)
        cs_xc = col_center_points[batch_idx].long() # (NumCol, 1)
        rs_yc, rs_sorted_idx = torch.sort(rs_yc, descending=False, dim=0) # sort (NumRow, 1)
        row_segm_logits_pb = row_segm_logits[batch_idx][rs_sorted_idx[:,0]] # sort (NumRow, H, W)
        cs_xc, cs_sorted_idx = torch.sort(cs_xc, descending=False, dim=0) # sort (NumCol, 1)
        col_segm_logits_pb = col_segm_logits[batch_idx][cs_sorted_idx[:,0]] # sort (NumCol, H, W)

        # parse col line segmentation
        _, col_line_index = col_segm_logits_pb.max(dim=2) # (NumCol, H), (NumCol, H)
        col_segm_map = torch.zeros_like(col_segm_logits_pb) # (NumCol, H, W)
        col_segm_map = col_segm_map.scatter(2, col_line_index[:, :, None].expand_as(col_segm_map), 1.) # (NumCol, H, W)
        col_segm_map[col_segm_logits_pb.sigmoid() <= score_threshold] = 0. # remove background
        col_segm_map = F.max_pool2d(col_segm_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/ 2)) # blur

        # parse row line segmentation
        _, row_line_index = row_segm_logits_pb.max(dim=1) # (NumRow, W), (NumRow, W)
        row_segm_map = torch.zeros_like(row_segm_logits_pb) # (NumRow, H, W)
        row_segm_map = row_segm_map.scatter(1, row_line_index[:, None, :].expand_as(row_segm_map), 1.) # (NumRow, H, W)
        row_segm_map[row_segm_logits_pb.sigmoid() <= score_threshold] = 0. # remove background
        row_segm_map = F.max_pool2d(row_segm_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/ 2)) # blur

        # parse the poly bbox
        num_rows = rs_yc.size(0)
        num_cols = cs_xc.size(0)
        grid_polys = list()
        for row_idx in range(num_rows-1):
            for col_idx in range(num_cols-1):
                x1_segm_map = col_segm_map[col_idx] # (H, W)
                y1_segm_map = row_segm_map[row_idx] # (H, W)
                x2_segm_map = col_segm_map[col_idx+1] # (H, W)
                y2_segm_map = row_segm_map[row_idx+1] # (H, W)

                # left top coordinate
                lt_segm_map = x1_segm_map + y1_segm_map # (H, W)
                valid_x1 = max(0, int(cs_xc[col_idx, 0] // stride_w - radius))
                valid_y1 = max(0, int(rs_yc[row_idx, 0] // stride_h - radius))
                valid_x2 = int(cs_xc[col_idx, 0] // stride_w + radius)
                valid_y2 = int(rs_yc[row_idx, 0] // stride_h + radius)
                valid_mask = torch.zeros_like(lt_segm_map)
                valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                lt_segm_map = lt_segm_map * valid_mask
                y_lt, x_lt = torch.where(lt_segm_map==2)
                if len(y_lt) > 0 and len(x_lt) > 0:
                    x_lt = int(x_lt.float().mean())
                    y_lt = int(y_lt.float().mean())
                else:
                    x_lt = int(cs_xc[col_idx, 0] // stride_w)
                    y_lt = int(rs_yc[row_idx, 0] // stride_h)

                # right top coordinate
                rt_segm_map = x2_segm_map + y1_segm_map # (H, W)
                valid_x1 = max(0, int(cs_xc[col_idx+1, 0] // stride_w - radius))
                valid_y1 = max(0, int(rs_yc[row_idx, 0] // stride_h - radius))
                valid_x2 = int(cs_xc[col_idx+1, 0] // stride_w + radius)
                valid_y2 = int(rs_yc[row_idx, 0] // stride_h + radius)
                valid_mask = torch.zeros_like(rt_segm_map)
                valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                rt_segm_map = rt_segm_map * valid_mask
                y_rt, x_rt = torch.where(rt_segm_map==2)
                if len(y_rt) > 0 and len(x_rt) > 0:
                    x_rt = int(x_rt.float().mean())
                    y_rt = int(y_rt.float().mean())
                else:
                    x_rt = int(cs_xc[col_idx+1, 0] // stride_w)
                    y_rt = int(rs_yc[row_idx, 0] // stride_h)

                # right bottom coordinate
                rb_segm_map = x2_segm_map + y2_segm_map # (H, W)
                valid_x1 = max(0, int(cs_xc[col_idx+1, 0] // stride_w - radius))
                valid_y1 = max(0, int(rs_yc[row_idx+1, 0] // stride_h - radius))
                valid_x2 = int(cs_xc[col_idx+1, 0] // stride_w + radius)
                valid_y2 = int(rs_yc[row_idx+1, 0] // stride_h + radius)
                valid_mask = torch.zeros_like(rb_segm_map)
                valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                rb_segm_map = rb_segm_map * valid_mask
                y_rb, x_rb = torch.where(rb_segm_map==2)
                if len(y_rb) > 0 and len(x_rb) > 0:
                    x_rb = int(x_rb.float().mean())
                    y_rb = int(y_rb.float().mean())
                else:
                    x_rb = int(cs_xc[col_idx+1, 0] // stride_w)
                    y_rb = int(rs_yc[row_idx+1, 0] // stride_h)

                # left bottom coordinate
                lb_segm_map = x1_segm_map + y2_segm_map # (H, W)
                valid_x1 = max(0, int(cs_xc[col_idx, 0] // stride_w - radius))
                valid_y1 = max(0, int(rs_yc[row_idx+1, 0] // stride_h - radius))
                valid_x2 = int(cs_xc[col_idx, 0] // stride_w + radius)
                valid_y2 = int(rs_yc[row_idx+1, 0] // stride_h + radius)
                valid_mask = torch.zeros_like(lb_segm_map)
                valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                lb_segm_map = lb_segm_map * valid_mask
                y_lb, x_lb = torch.where(lb_segm_map==2)
                if len(y_lb) > 0 and len(x_lb) > 0:
                    x_lb = int(x_lb.float().mean())
                    y_lb = int(y_lb.float().mean())
                else:
                    x_lb = int(cs_xc[col_idx, 0] // stride_w)
                    y_lb = int(rs_yc[row_idx+1, 0] // stride_h)

                grid_polys.append([x_lt, y_lt, x_rt, y_rt, x_rb, y_rb, x_lb, y_lb])

        if len(grid_polys) == 0:
            grid_polys.append([0, 0, 0, 0, 0, 0, 0, 0])
            num_cols = 2
            num_rows = 2

        grid_polys = torch.tensor(grid_polys, dtype=torch.float, device=row_segm_logits[0].device)
        grid_polys[:, 0::2] *= stride_w
        grid_polys[:, 1::2] *= stride_h

        batch_grid_bboxes.append(grid_polys)
        batch_num_cols.append(num_cols-1)
        batch_num_rows.append(num_rows-1)
    return batch_grid_bboxes, batch_num_rows, batch_num_cols


def cal_merge_acc(preds, labels, masks):
    '''calculate the accuracy of merger
    preds: (B, N, H, W)
    labels: (B, N, H, W)
    masks: (B, H, W)
    '''
    B, N, _, _ = preds.shape
    preds = (preds * masks[:, None]).reshape(B, N, -1) # (B, N, H*W)
    labels = (labels * masks[:, None]).reshape(B, N, -1) # (B, N, H*W)
    correct_num = ((preds == labels).min(-1)[0].float() * labels.max(-1)[0]).sum()
    total_num = labels.max(-1)[0].sum()
    return correct_num / (total_num + 1e-5)


def poly2bbox(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x1,y1,x2,y2,x3,y3,x4,y4]
        
    Returns:
        obbs (torch.Tensor): [x1,y1,x2,y2]
    """

    polys = polys.reshape(-1, 4, 2) # (N, 4, 2)
    x1 = polys[:, :, 0].min(-1)[0] # (N)
    y1 = polys[:, :, 1].min(-1)[0] # (N)
    x2 = polys[:, :, 0].max(-1)[0] # (N)
    y2 = polys[:, :, 1].max(-1)[0] # (N)
    polys = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)
    return polys


def poly2obb(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x_lt,y_lt,x_rt,y_rt,x_rb,y_rb,x_lb,y_lb]
        
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    w1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    w2 = torch.sqrt(
        torch.pow(pt4[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt4[..., 1] - pt3[..., 1], 2))
    h1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt4[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt4[..., 1], 2))
    h2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))

    edge1 = (w1 + w2) / 2
    edge2 = (h1 + h2) / 2
    
    angles_1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles_2 = torch.atan2((pt3[..., 1] - pt4[..., 1]),
                          (pt3[..., 0] - pt4[..., 0]))

    angles = (angles_1 + angles_2) / 2

    angles = (angles + np.pi / 2) % np.pi - np.pi / 2
    x_ctr = (pt1[..., 0] + pt2[..., 0] + pt3[..., 0] + pt4[..., 0]) / 4.0
    y_ctr = (pt1[..., 1] + pt2[..., 1] + pt3[..., 1] + pt4[..., 1]) / 4.0
    width = edge1
    height = edge2

    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def obb2poly(obbs):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        
    Returns:
        polys (torch.Tensor): [x_lt,y_lt,x_rt,y_rt,x_rb,y_rb,x_lb,y_lb]
    """
    x1 = torch.cos(obbs[:, 4]) * (-obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 0]
    x2 = torch.cos(obbs[:, 4]) * (obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 0]
    x3 = torch.cos(obbs[:, 4]) * (-obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 0]
    x4 = torch.cos(obbs[:, 4]) * (obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 0]
    y1 = torch.sin(obbs[:, 4]) * (-obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 1]
    y2 = torch.sin(obbs[:, 4]) * (obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 1]
    y3 = torch.sin(obbs[:, 4]) * (-obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 1]
    y4 = torch.sin(obbs[:, 4]) * (obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 1]
    polys = torch.stack([x1,y1,x2,y2,x4,y4,x3,y3], dim=-1)
    return polys