import cv2
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from functools import reduce
import operator
import math
from PIL import Image, ImageDraw, ImageFont


def process_layout(score, index, use_score=False, is_merge=True, score_threshold=0.5):
    if use_score:
        if is_merge:
            y, x = torch.where(score < score_threshold)
            index[y, x] = index.max() + 1
        else:
            y, x = torch.where(score < score_threshold)
            index[y, x] = torch.arange(index.max() + 1, index.max() + 1 + len(y)).to(index.device, index.dtype)

    layout = torch.full_like(index, -1)
    layout_mask = torch.full_like(index, -1)
    nrow, ncol = score.shape
    for cell_id in range(max(nrow * ncol, index.max() + 1)):
        if layout_mask.min() != -1:
            break
        crow, ccol = torch.where(layout_mask == layout_mask.min())
        ccol = ccol[crow == crow.min()].min()
        crow = crow.min()
        id = index[crow, ccol]
        h, w = torch.where(index == id)
        if h.shape[0] == 1 or w.shape[0] == 1: # single
            layout_mask[h, w] = 1
            layout[h, w] = cell_id
            continue
        else:
            h_min = h.min()
            h_max = h.max()
            w_min = w.min()
            w_max = w.max()
            if torch.all(index[h_min:h_max+1, w_min:w_max+1] == id):
                layout_mask[h_min:h_max+1, w_min:w_max+1] = 1
                layout[h_min:h_max+1, w_min:w_max+1] = cell_id
            else:
                lf_row = crow
                lf_col = ccol
                col_mem = -1
                for col_ in range(lf_col, w_max + 1):
                    if index[lf_row, col_] == id:
                        layout_mask[lf_row, col_] = 1
                        layout[lf_row, col_] = cell_id
                        col_mem = col_
                    else:
                        break
                for row_ in range(lf_row + 1, h_max + 1):
                    if torch.all(index[row_, lf_col: col_mem + 1] == id):
                        layout_mask[row_, lf_col: col_mem + 1] = 1
                        layout[row_, lf_col: col_mem + 1] = cell_id
                    else:
                        break
    return layout


def layout2spans(layout):
    rows, cols = layout.shape[-2:]
    cells_span = list()
    for cell_id in range(rows * cols):
        cell_positions = np.argwhere(layout == cell_id)
        if len(cell_positions) == 0:
            continue
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return cells_span


def trans2cellbbox(grid_bboxes):
    '''
    trans the input grid bboxes (N,4) to cell bbox
    '''
    grid_bboxes = np.array(grid_bboxes).reshape(-1, 2)
    x1 = int(grid_bboxes[:, 0].min())
    y1 = int(grid_bboxes[:, 1].min())
    x2 = int(grid_bboxes[:, 0].max())
    y2 = int(grid_bboxes[:, 1].max())
    return [x1, y1, x2, y2]


def trans2cellpoly(grid_polys):
    '''
    trans the input grid polys (N,8) to cell poly
    clock-wise
    '''
    grid_polys = np.array(grid_polys).reshape(-1, 2).tolist()
    grid_polys = [tuple(item) for item in grid_polys]
    grid_polys = list(set(grid_polys))
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), grid_polys), [len(grid_polys)] * 2))
    grid_polys = sorted(grid_polys, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    grid_polys = [list(item) for item in grid_polys]
    return grid_polys


def parse_cells(layout, spans, grid_bboxes, grid_polys):

    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout.reshape(-1)==cell_id)
        valid_grid_bboxes = grid_bboxes[cell_positions[:, 0]]
        valid_grid_polys = grid_polys[cell_positions[:, 0]]
        cell_bbox = trans2cellbbox(valid_grid_bboxes)
        cell_poly = trans2cellpoly(valid_grid_polys)

        span = spans[cell_id]

        cell = dict(
            bbox=cell_bbox,
            segmentation=[cell_poly],
            col_start_idx=int(span[0]),
            row_start_idx=int(span[1]),
            col_end_idx=int(span[2]),
            row_end_idx=int(span[3])
        )
        cells.append(cell)

    return cells


def parse_layout(mg_logits, num_rows, num_cols, score_threshold=0.5):
    num_grids = int(num_rows) * int(num_cols)
    mg_probs = mg_logits[:num_grids, :int(num_rows), :int(num_cols)].sigmoid() # (N, H, W)
    _, indices = (mg_probs > score_threshold).float().max(dim=0) # (H, W)
    values, _ = mg_probs.max(dim=0) # (H, W)
    layout = process_layout(values, indices, use_score=True, is_merge=False, score_threshold=score_threshold)
    layout = process_layout(values, layout)
    layout = layout.cpu().numpy()
    spans = layout2spans(layout)
    return layout, spans


def parse_grid_bboxes(row_center_points, row_segm_logits, col_center_points, col_segm_logits,\
    stride_w, stride_h, score_threshold=0.5, kernel_size=3, radius=1000):
    '''
        parse the start bboxes and segmentation results to table grid bboxes
    '''

    batch_grid_bboxes = []
    batch_num_rows = []
    batch_num_cols = []
    batch_col_segm_map = []
    batch_row_segm_map = []
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
        batch_col_segm_map.append(col_segm_map)

        # parse row line segmentation
        _, row_line_index = row_segm_logits_pb.max(dim=1) # (NumRow, W), (NumRow, W)
        row_segm_map = torch.zeros_like(row_segm_logits_pb) # (NumRow, H, W)
        row_segm_map = row_segm_map.scatter(1, row_line_index[:, None, :].expand_as(row_segm_map), 1.) # (NumRow, H, W)
        row_segm_map[row_segm_logits_pb.sigmoid() <= score_threshold] = 0. # remove background
        row_segm_map = F.max_pool2d(row_segm_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/ 2)) # blur
        batch_row_segm_map.append(row_segm_map)

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
    return batch_grid_bboxes, batch_num_rows, batch_num_cols, batch_row_segm_map, batch_col_segm_map


def visualize(table, pred_result, prefix, stride=4, back_color=(255,255,255), text_color=(255,0,0),  
    font_size=10, font_path='/yrfs1/intern/zrzhang6/DocumentPretrain/dataprocess/process_rvlcdip/libs/simfang.ttf'):

    row_start_bboxes, row_center_points, row_segm_logits, \
            col_start_bboxes, col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols = pred_result

    # draw row center point
    image = copy.deepcopy(table['img'])
    cv2.imwrite(prefix+'_origin.png', image)
    for bbox in row_start_bboxes.cpu().numpy():
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        x1, y1, x2, y2 = [int(item) for item in bbox]
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        cv2.circle(image, (xc, yc), 5, color, -1)
    cv2.imwrite(prefix+'_row_point.png', image)

    # draw col center point
    image = copy.deepcopy(table['img'])
    for bbox in col_start_bboxes.cpu().numpy():
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        x1, y1, x2, y2 = [int(item) for item in bbox]
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        cv2.circle(image, (xc, yc), 5, color, -1)
    cv2.imwrite(prefix+'_col_point.png', image)

    # draw grid polys
    grid_polys, *_, row_segm_map, col_segm_map = parse_grid_bboxes([row_center_points], [row_segm_logits], \
        [col_center_points], [col_segm_logits], stride, stride)
    image = copy.deepcopy(table['img'])
    for poly in grid_polys[0].cpu().numpy():
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        poly = np.array(poly).reshape(-1,1,2).astype('int')
        cv2.polylines(image, [poly], True, color, 2)
    cv2.imwrite(prefix+'_grid_poly.png', image)

    # draw grid bboxes
    grid_bboxes = grid_polys[0].reshape(-1, 4, 2) # (N, 4, 2)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = [torch.stack([x1, y1, x2, y2], dim=-1)] # (N, 4)
    image = copy.deepcopy(table['img'])
    for bbox in grid_bboxes[0].cpu().numpy():
        x1, y1, x2, y2 = [int(item) for item in bbox]
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite(prefix+'_grid_bbox.png', image)

    # resize the segm map
    image = copy.deepcopy(table['img'])
    image_h, image_w = image.shape[:2]
    valid_h, valid_w = int(image_h / stride), int(image_w / stride)

    row_segm_map = row_segm_map[0][:, :valid_h, :valid_w]
    row_segm_map = F.interpolate(row_segm_map.unsqueeze(1).float(), size=(image_h, image_w), mode='nearest').squeeze(1)
    row_segm_map = row_segm_map.cpu().numpy()

    col_segm_map = col_segm_map[0][:, :valid_h, :valid_w]
    col_segm_map = F.interpolate(col_segm_map.unsqueeze(1).float(), size=(image_h, image_w), mode='nearest').squeeze(1)
    col_segm_map = col_segm_map.cpu().numpy()

    # draw row line segm
    image = copy.deepcopy(table['img'])
    for segm_map in row_segm_map:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image[segm_map==1.] = color
    cv2.imwrite(prefix+'-row_line_segm.png', image)

    # draw each row line segm
    for idx, segm_map in enumerate(row_segm_map):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = copy.deepcopy(table['img'])
        image[segm_map==1.] = color
        cv2.imwrite(prefix+'-%02d_row_line_segm.png'%idx, image)

    # draw col line segm
    image = copy.deepcopy(table['img'])
    for segm_map in col_segm_map:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image[segm_map==1.] = color
    cv2.imwrite(prefix+'-col_line_segm.png', image)

    # draw each col line segm
    for idx, segm_map in enumerate(col_segm_map):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = copy.deepcopy(table['img'])
        image[segm_map==1.] = color
        cv2.imwrite(prefix+'-%02d_col_line_segm.png'%idx, image)
    
    # parse table layout
    layout, spans = parse_layout(mg_logits, num_rows, num_cols)
    cells = parse_cells(layout, spans, grid_bboxes[0].cpu().numpy(), grid_polys[0].cpu().numpy())
    
    # draw cell bbox
    image = copy.deepcopy(table['img'])
    for cell in cells:
        x1, y1, x2, y2 = [int(item) for item in cell['bbox']]
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # draw layout info
    image = Image.fromarray(image)
    draw_img = ImageDraw.Draw(image)
    for cell in cells:
        x1, y1, *_ = [int(item) for item in cell['bbox']]
        txt = '%d,%d,%d,%d' % (cell['col_start_idx'], cell['row_start_idx'], \
            cell['col_end_idx'], cell['row_end_idx'])
        num_txt = len(txt) - 3
        # draw text backgroud
        x2 = x1 + num_txt * font_size
        y2 = y1 + font_size
        draw_img.polygon([x1,y1,x2,y1,x2,y2,x1,y2], fill=back_color)
        # draw text foreground
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        draw_img.text([x1, y1], txt, fill=text_color, font=font)
    cv2.imwrite(prefix+'_cell_bbox.png', np.array(image))

    # draw cell poly
    image = copy.deepcopy(table['img'])
    for cell in cells:
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        poly = np.array(cell['segmentation']).reshape(-1,1,2).astype('int')
        cv2.polylines(image, [poly], True, color, 2)
    # draw layout info
    image = Image.fromarray(image)
    draw_img = ImageDraw.Draw(image)
    for cell in cells:
        x1, y1, *_ = [int(item) for item in cell['bbox']]
        txt = '%d,%d,%d,%d' % (cell['col_start_idx'], cell['row_start_idx'], \
            cell['col_end_idx'], cell['row_end_idx'])
        num_txt = len(txt) - 3
        # draw text backgroud
        x2 = x1 + num_txt * font_size
        y2 = y1 + font_size
        draw_img.polygon([x1,y1,x2,y1,x2,y2,x1,y2], fill=back_color)
        # draw text foreground
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        draw_img.text([x1, y1], txt, fill=text_color, font=font)
    cv2.imwrite(prefix+'_cell_poly.png', np.array(image))