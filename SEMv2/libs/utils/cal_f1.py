import torch
import tqdm
import copy
import Polygon
import numpy as np
from .visualize import parse_grid_bboxes, trans2cellbbox, trans2cellpoly
from .scitsr.eval import json2Relations, eval_relations


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


def parse_cells(layout, spans, row_center_points, row_segm_logits,\
        col_center_points, col_segm_logits):

    grid_polys, *_ = parse_grid_bboxes([row_center_points], [row_segm_logits], \
        [col_center_points], [col_segm_logits], stride_w=4, stride_h=4, score_threshold=0.25)

    grid_polys = grid_polys[0]
    grid_bboxes = grid_polys.reshape(-1, 4, 2) # (N, 4, 2)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)

    grid_polys = grid_polys.cpu().numpy().astype('int64')
    grid_bboxes = grid_bboxes.cpu().numpy().astype('int64')
    
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


def extend_cell_lines(cells, lines):
    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    lines = copy.deepcopy(lines)

    cells_poly = [segmentation_to_polygon(item['segmentation']) for item in cells]
    lines_poly = [segmentation_to_polygon(item['segmentation']) for item in lines]

    cells_lines = [[] for _ in range(len(cells))]

    for line_idx, line_poly in enumerate(lines_poly):
        if line_poly.area() == 0:
            continue
        line_area = line_poly.area()
        max_overlap = 0
        max_overlap_idx = None
        for cell_idx, cell_poly in enumerate(cells_poly):
            overlap = (cell_poly & line_poly).area() / line_area
            if overlap > max_overlap:
                max_overlap_idx = cell_idx
                max_overlap = overlap
        if max_overlap > 0:
            cells_lines[max_overlap_idx].append(line_idx)
    lines_y1 = [segmentation_to_bbox(item['segmentation'])[1] for item in lines]
    cells_lines = [sorted(item, key=lambda idx: lines_y1[idx]) for item in cells_lines]

    for cell, cell_lines in zip(cells, cells_lines):
        transcript = []
        for idx in cell_lines:
            transcript.extend(lines[idx]['transcript'])
        cell['transcript'] = transcript


def segmentation_to_bbox(segmentation):
    x1 = min([min([pt[0] for pt in contour]) for contour in segmentation])
    y1 = min([min([pt[1] for pt in contour]) for contour in segmentation])
    x2 = max([max([pt[0] for pt in contour]) for contour in segmentation])
    y2 = max([max([pt[1] for pt in contour]) for contour in segmentation])
    return [x1, y1, x2, y2]


def cal_cell_spans(table):
    layout = table['layout']
    num_cells = len(table['cells'])
    cells_span = list()
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout == cell_id)
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return cells_span


def pred_result_to_table(table, pred_result):
    # gt ocr result
    lines = [dict(segmentation=cell['segmentation'], transcript=cell['transcript']) for cell in table['cells'] if 'bbox' in cell.keys()]

    row_center_points, row_segm_logits, \
            col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols = pred_result
    
    layout, spans = parse_layout(mg_logits, num_rows, num_cols)
    cells = parse_cells(layout, spans, row_center_points, row_segm_logits,\
        col_center_points, col_segm_logits)
    extend_cell_lines(cells, lines)

    table = dict(
        layout=layout,
        cells=cells
    )
    
    return table


def table_to_relations(table):
    cell_spans = cal_cell_spans(table)
    contents = [''.join(cell['transcript']).split() for cell in table['cells']]
    relations = []
    for span, content in zip(cell_spans, contents):
        x1, y1, x2, y2 = span
        relations.append(dict(start_row=y1, end_row=y2, start_col=x1, end_col=x2, content=content))
    return dict(cells=relations)


def cal_f1(label, pred):
    label = json2Relations(label, splitted_content=True)
    pred = json2Relations(pred, splitted_content=True)
    precision, recall = eval_relations(gt=[label], res=[pred], cmp_blank=True)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return [precision, recall, f1]


def single_process(labels, preds):
    scores = dict()
    for key in tqdm.tqdm(labels.keys()):
        pred = preds.get(key, '')
        label = labels.get(key, '')
        score = cal_f1(label, pred)
        scores[key] = score
    return scores


def _worker(labels, preds,  keys, result_queue):
    for key in keys:
        label = labels.get(key, '')
        pred = preds.get(key, '')
        score = cal_f1(label, pred)
        result_queue.put((key, score))


def multi_process(labels, preds, num_workers):
    import multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    keys = list(labels.keys())
    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker,
            args=(
                labels,
                preds,
                keys[worker_idx::num_workers],
                result_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    scores = dict()
    tq = tqdm.tqdm(total=len(keys))
    for _ in range(len(keys)):
        key, val = result_queue.get()
        scores[key] = val
        P, R, F1 = (100 * np.array(list(scores.values()))).mean(0).tolist()
        tq.set_description('P: %.2f, R: %.2f, F1: %.2f' % (P, R, F1), False)
        tq.update()
    
    return scores


def evaluate_f1(labels, preds, num_workers=0):
    preds = {idx: pred for idx, pred in enumerate(preds)}
    labels = {idx: label for idx, label in enumerate(labels)}
    if num_workers == 0:
        scores = single_process(labels, preds)
    else:
        scores = multi_process(labels, preds, num_workers)
    sorted_idx = sorted(list(range(len(list(scores)))), key=lambda idx: list(scores.keys())[idx])
    scores = [scores[idx] for idx in sorted_idx]
    return scores