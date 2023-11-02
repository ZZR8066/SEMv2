import numpy as np
import Polygon
import copy


# parse relation adjacency matrix
def parse_relation_from_table(table, iou_threshold=0.5):
    if table['is_wireless']:
        line_polys = table['line']
    else:
        line_polys = table['cell']

    # parse row relation adjacency matrix
    row_adj = np.identity(len(line_polys), dtype=np.int64)
    row_polys = table['row']
    for row_poly in row_polys:
        same_row_idxs = []
        row_polygon = Polygon.Polygon(row_poly)
        for idx, line_poly in enumerate(line_polys):
            line_polygon = Polygon.Polygon(line_poly)
            iou = (row_polygon & line_polygon).area() / min(row_polygon.area(), line_polygon.area())
            if iou >= iou_threshold:
                same_row_idxs.append(idx)
        # map to row relation adjacency matrix
        for i in same_row_idxs:
            for j in same_row_idxs:
                row_adj[i,j] = 1
    table['row_adj'] = row_adj

    # parse col relation adjacency matrix
    col_adj = np.identity(len(line_polys), dtype=np.int64)
    col_polys = table['col']
    for col_poly in col_polys:
        same_col_idxs = []
        col_polygon = Polygon.Polygon(col_poly)
        for idx, line_poly in enumerate(line_polys):
            line_polygon = Polygon.Polygon(line_poly)
            iou = (col_polygon & line_polygon).area() / min(col_polygon.area(), line_polygon.area())
            if iou >= iou_threshold:
                same_col_idxs.append(idx)
        # map to col relation adjacency matrix
        for i in same_col_idxs:
            for j in same_col_idxs:
                col_adj[i,j] = 1
    table['col_adj'] = col_adj

    # parse cell relation adjacency matrix
    cell_adj= np.array((row_adj + col_adj)==2, dtype=np.int64)
    table['cell_adj'] = cell_adj

    return table


# get spanning cells idx
def get_span_cells(row_adj, col_adj):
    row_span_indice = []
    for row_idx, row in enumerate(row_adj):
        idx_r = list(np.where(row == 1)[0])
        if len(idx_r) > 2:
            idx_r.remove(row_idx)
            for idx1 in idx_r: # justify this cell is spanning cell or not
                for idx2 in idx_r:
                    if row_adj[idx1, idx2] != 1:
                        row_span_indice.append(row_idx)

    col_span_indice = []
    for col_idx, col in enumerate(col_adj):
        idx_c = list(np.where(col == 1)[0])
        if len(idx_c) > 2:
            idx_c.remove(col_idx)
            for idx1 in idx_c: # justify this cell is spanning cell or not
                for idx2 in idx_c:
                    if col_adj[idx1, idx2] != 1:
                        col_span_indice.append(col_idx)
    span_text_indice = list(set(row_span_indice + col_span_indice))
    row_span_text_indice = list(set(row_span_indice))
    col_span_text_indice = list(set(col_span_indice))
    return span_text_indice, row_span_text_indice, col_span_text_indice


def get_shared_line(adj_mat, adj_cell, table, span_index):
    if table['is_wireless']:
        text_box = table['line']
    else:
        text_box = table['cell']

    all_index = list(range(len(text_box)))
    for sidx in span_index:
        all_index.remove(sidx)
    
    adj_mat_wo_span = adj_mat[all_index][:,all_index]
    adj_cell_wo_span = adj_cell[all_index][:,all_index]

    text_box_wo_span = [text_box[idx_] for idx_ in range(len(text_box)) if idx_ not in span_index]
    
    neglect = []
    text_share_all = []
    for ridx, adj in enumerate(adj_mat_wo_span):
        if ridx not in neglect:
            text_idx = adj.nonzero()[0]
            text_share = []
            neglect.extend(text_idx)
            neglect_share_cell = []
            for tidx in text_idx:
                if tidx not in neglect_share_cell:
                    text_idx_c = adj_cell_wo_span[tidx].nonzero()[0]
                    neglect_share_cell.extend(text_idx_c)
                    text_share.append([text_box_wo_span[idx_] for idx_ in text_idx_c])
            text_share_all.append(text_share)
    
    return text_share_all


def get_shared_line_id(adj_mat, adj_cell, span_index):   
    neglect = []
    text_share_all = []
    for ridx, adj in enumerate(adj_mat):
        if ridx in span_index:
            continue
        if ridx not in neglect:
            text_idx = adj.nonzero()[0]
            # remove span index
            text_idx = [idx for idx in text_idx if idx not in span_index]
            text_share = []
            neglect.extend(text_idx)
            neglect_share_cell = []
            for tidx in text_idx:
                if tidx not in neglect_share_cell:
                    text_idx_c = adj_cell[tidx].nonzero()[0]
                    neglect_share_cell.extend(text_idx_c)
                    for idx_ in text_idx_c:
                        text_share.append(idx_)
            text_share_all.append(text_share)

    return text_share_all


def sort_shared_line(share_text_id_row, shared_text_row, share_text_id_col, shared_text_col):
    # sort rows from top to down
    row_locs = []
    for row_text in shared_text_row:
        points_ = np.vstack([np.vstack(itm) for itm in row_text])
        row_loc_ = np.mean(points_, axis=0)[1]
        row_locs.append(row_loc_)
    row_index = np.argsort(row_locs)
    share_text_id_row = [share_text_id_row[idx_] for idx_ in row_index]
    shared_text_row = [shared_text_row[idx_] for idx_ in row_index]

    # sort cols from left to right
    col_locs = []
    for col_text in shared_text_col:
        points_ = np.vstack([np.vstack(itm) for itm in col_text])
        col_loc_ = np.mean(points_, axis=0)[0]
        col_locs.append(col_loc_)
    col_index = np.argsort(col_locs)
    share_text_id_col = [share_text_id_col[idx_] for idx_ in col_index]
    shared_text_col = [shared_text_col[idx_] for idx_ in col_index]

    return share_text_id_row, shared_text_row, share_text_id_col, shared_text_col


def format_layout(layout):
    new_layout = np.full_like(layout, -1)
    row_nums, col_nums = layout.shape
    cell_id = 0
    for row_id in range(row_nums):
        for col_id in range(col_nums):
            if new_layout[row_id, col_id] == -1:
                y, x = np.where(layout==layout[row_id, col_id])
                new_layout[y, x] = cell_id
                cell_id += 1
    assert new_layout.min() >= 0
    return new_layout


def parse_gt_label(cell_adj, row_adj, col_adj, shared_row_line_ids, shared_col_line_ids):
    num_row = len(shared_row_line_ids)
    num_col = len(shared_col_line_ids)
    if num_row == 0 or num_col == 0:
        table = dict(
            layout=np.zeros((1,1), dtype=np.int64).tolist(),
            cells=[dict(
                col_start_idx=0,
                row_start_idx=0,
                col_end_idx=0,
                row_end_idx=0,
                transcript='0'
            )]
        )
        return table

    layout = np.arange(int(num_row*num_col)).reshape(num_row, num_col)
    start_id = int(num_row*num_col)

    neglect = [] # passed assigned cell ids
    assign_text_id = dict() # save assigned cell ids
    for index, adj in enumerate(cell_adj):
        if index in neglect: # justify assign or not
            continue
        cell_ids = adj.nonzero()[0]
        neglect.extend(cell_ids)

        span_row_ids = []
        span_col_ids = []

        # find all span row line ids
        span_row_line_ids = []
        for ids in cell_ids:
            span_row_line_ids.extend(row_adj[ids].nonzero()[0])
        span_row_line_ids = list(set(span_row_line_ids))
        for row_id, text_ids in enumerate(shared_row_line_ids):
            for idx in text_ids:
                if idx in span_row_line_ids:
                    span_row_ids.append(row_id)
                    break
        
        # find all span col line ids
        span_col_line_ids = []
        for ids in cell_ids:
            span_col_line_ids.extend(col_adj[ids].nonzero()[0])
        span_col_line_ids = list(set(span_col_line_ids))
        for col_id, text_ids in enumerate(shared_col_line_ids):
            for idx in text_ids:
                if idx in span_col_line_ids:
                    span_col_ids.append(col_id)
                    break
        
        start_row = min(span_row_ids)
        end_row = max(span_row_ids)
        start_col = min(span_col_ids)
        end_col = max(span_col_ids)
        layout[start_row:end_row+1, start_col:end_col+1] = start_id + index

        sorted(cell_ids)
        cell_ids = [str(item) for item in cell_ids]
        span = '%d-%d-%d-%d' % (start_col, start_row, end_col, end_row)
        assign_text_id[span] = '-'.join(cell_ids)

    layout = format_layout(layout)
    # cells
    cells = list()
    num_cells = layout.max() + 1
    for cell_id in range(num_cells):
        ys, xs = np.split(np.argwhere(layout==cell_id), 2, 1)
        start_row = ys.min()
        end_row = ys.max()
        start_col = xs.min()
        end_col = xs.max()
        span = '%d-%d-%d-%d' % (start_col, start_row, end_col, end_row)
        if span in assign_text_id.keys():
            transcript = assign_text_id[span]
        else:
            transcript = ''
        cell = dict(
            col_start_idx=int(start_col),
            row_start_idx=int(start_row),
            col_end_idx=int(end_col),
            row_end_idx=int(end_row),
            transcript=transcript
        )
        cells.append(cell)
        
    table = dict(
        layout=layout.tolist(),
        cells=cells
    )
    return table


def extend_text_lines(cells, lines):
    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    lines = copy.deepcopy(lines)

    cells_poly = [segmentation_to_polygon(item['segmentation']) for item in cells]
    # lines_poly = [segmentation_to_polygon(item['segmentation']) for item in lines]
    lines_poly = [segmentation_to_polygon([item]) for item in lines]

    assign_ids = dict()
    for idx in range(len(cells_poly)):
        assign_ids[idx] = list()

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
            if overlap > 0.99:
                break
        if max_overlap > 0:
            assign_ids[max_overlap_idx].append(line_idx)
    
    for idx, value in assign_ids.items():
        sorted(value)
        value = [str(item) for item in value]
        cells[idx]['transcript'] = '-'.join(value)
        
    return cells


def update_gt_label(gt_label, table):
    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    if table['is_wireless']:
        return gt_label
    else:
        cells = gt_label['cells']
        lines = copy.deepcopy(table['line'])
        lines_poly = [segmentation_to_polygon([item]) for item in lines]

        assign_cells = list()
        for _idx, cell in enumerate(cells):
            if len(cell['transcript']) > 0:
                cell_ids = [int(idx) for idx in cell['transcript'].split('-')]
                cells_poly = [table['cell'][idx] for idx in cell_ids]
                cells_poly = [segmentation_to_polygon([item]) for item in cells_poly]
                assign_cells.append(cells_poly)
            else:
                cells_poly = [[[0,0],[0,0],[0,0],[0,0]]]
                cells_poly = [segmentation_to_polygon([item]) for item in cells_poly]
                assign_cells.append(cells_poly)
    
    # assign line idx to cells
    assign_line_ids = [[] for _ in range(len(assign_cells))]
    for line_idx, line_poly in enumerate(lines_poly):
        if line_poly.area() == 0:
            continue
        line_area = line_poly.area()
        max_overlap = 0
        max_overlap_idx = None
        for _idx, cells_poly in enumerate(assign_cells):
            for cell_poly in cells_poly:
                overlap = (cell_poly & line_poly).area() / line_area
                if overlap > max_overlap:
                    max_overlap_idx = _idx
                    max_overlap = overlap
                if overlap > 0.99:
                    break
        if max_overlap > 0:
            assign_line_ids[max_overlap_idx].append(line_idx)

    # rewrite line idx to transcript
    for idx, value in enumerate(assign_line_ids):
        sorted(value)
        value = [str(item) for item in value]
        cells[idx]['transcript'] = '-'.join(value)
    
    gt_label['cells'] = cells
    return gt_label