import cv2
import copy
import Polygon
import numpy as np


def cal_mean_lr(optimizer):
    lrs = [group['lr'] for group in optimizer.param_groups]
    return sum(lrs)/len(lrs)


def cal_pr_f1(pr_info):
    precision = pr_info[0] / pr_info[1]
    recall = pr_info[0] / pr_info[2]
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1


def match_segment_spans(segments, spans):
    matched_segments = list()
    matched_spans = list()

    for segment_idx, segment in enumerate(segments):
        for span_idx, span in enumerate(spans):
            if span_idx not in matched_spans:
                if (segment >= span[0]) and (segment < span[1]):
                    matched_segments.append(segment_idx)
                    matched_spans.append(span_idx)
    
    return matched_segments, matched_spans


def find_unmatch_segment_spans(segments, spans):
    unmatched_segments = list()
    for segment_idx, segment in enumerate(segments):
        matched = False
        for span in spans:
            if (segment >= span[0]) and (segment < span[1]):
                matched = True
                break
        if not matched:
            unmatched_segments.append(segment_idx)
        
    return unmatched_segments


def parse_layout(spans, num_rows, num_cols):
    layout = np.full([num_rows, num_cols], -1, dtype=np.int)
    cell_count = 0
    for x1, y1, x2, y2 in spans:
        layout[y1:y2+1, x1:x2+1] = cell_count
        cell_count += 1

    cells_id = list()
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            cell_id = layout[row_idx, col_idx]
            if cell_id in cells_id:
                layout[row_idx, col_idx] = cells_id.index(cell_id)
            else:
                layout[row_idx, col_idx] = len(cells_id)
                cells_id.append(cell_id)
    return layout


def parse_cells(layout, spans, row_segments, col_segments):
    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout == cell_id)
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        x1 = col_segments[x1]
        x2 = col_segments[x2+1]
        y1 = row_segments[y1]
        y2 = row_segments[y2+1]
        cell = dict(
            segmentation=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        )
        cells.append(cell)
    for span in spans:
        cell_id = layout[span[1], span[0]]
        cells[cell_id]['transcript'] = 'None'
    return cells


def segmentation_to_bbox(segmentation):
    x1 = min([min([pt[0] for pt in contour]) for contour in segmentation])
    y1 = min([min([pt[1] for pt in contour]) for contour in segmentation])
    x2 = max([max([pt[0] for pt in contour]) for contour in segmentation])
    y2 = max([max([pt[1] for pt in contour]) for contour in segmentation])
    return [x1, y1, x2, y2]


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
            overlap = (cell_poly & line_poly).area()/line_area
            if overlap > max_overlap:
                max_overlap_idx = cell_idx
                max_overlap = overlap
        if max_overlap > 0:
            cells_lines[max_overlap_idx].append(line_idx)
    lines_y1 = [segmentation_to_bbox(item['segmentation'])[1] for item in lines]
    cells_lines = [sorted(item, key=lambda idx: lines_y1[idx]) for item in cells_lines]

    for cell, cell_lines in zip(cells, cells_lines):
        cell['lines_idx'] = cell_lines


def rerange_layout(table):
    layout = table['layout']
    cells = table['cells']
    valid_cells_id = list()
    for row_idx in range(layout.shape[0]):
        for col_idx in range(layout.shape[1]):
            cell_id = layout[row_idx, col_idx]
            if cell_id not in valid_cells_id:
                valid_cells_id.append(cell_id)
            layout[row_idx, col_idx] = valid_cells_id.index(cell_id)
    cells = [cells[cell_id] for cell_id in valid_cells_id]
    table['layout'] = layout
    table['cells'] = cells

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


def remove_repeat_rcs(table):
    layout = table['layout']
    head_rows = table['head_rows']
    body_rows = table['body_rows']
    while True:
        num_rows = layout.shape[0]
        num_cols = layout.shape[1]
        valid_rows_idx = list()
        valid_rows_key = list()

        for row_idx in range(num_rows):
            row = layout[row_idx, :]
            if len(np.unique(row)) == 1 and row_idx in body_rows: # remove repeated row
                continue
            row_key = ','.join([str(item) for item in row])
            if row_key not in valid_rows_key:
                valid_rows_idx.append(row_idx)
                valid_rows_key.append(row_key)

        valid_cols_idx = list()
        valid_cols_key = list()
        for col_idx in range(num_cols):
            col = layout[:, col_idx]
            if len(np.unique(col)) == 1: # remove repeated col
                continue
            col_key = ','.join([str(item) for item in col])
            if col_key not in valid_cols_key:
                valid_cols_idx.append(col_idx)
                valid_cols_key.append(col_key)
        if (len(valid_rows_idx) == num_rows) and (len(valid_cols_idx) == num_cols):
            break
        layout = layout[valid_rows_idx][:, valid_cols_idx]
        head_rows = [n_idx for n_idx, o_idx in enumerate(valid_rows_idx) if o_idx in head_rows]
        body_rows = [n_idx for n_idx, o_idx in enumerate(valid_rows_idx) if o_idx in body_rows]

    table['layout'] = layout
    table['head_rows'] = head_rows
    table['body_rows'] = body_rows
    rerange_layout(table)


def pred_result_to_table(pred_result):
    row_segments, col_segments, divide, spans = pred_result
    num_rows = len(row_segments) - 1
    num_cols = len(col_segments) - 1

    layout = parse_layout(spans, num_rows, num_cols)
    cells = parse_cells(layout, spans, row_segments, col_segments)
    head_rows = list(range(0, divide))
    body_rows = list(range(divide, num_rows))
    
    table = dict(
        layout=layout,
        head_rows=head_rows,
        body_rows=body_rows,
        cells=cells
    )

    # remove_repeat_rcs(table)
    
    return table


def is_simple_table(table):
    layout = table['layout']
    num_rows, num_cols = layout.shape
    if num_rows * num_cols == len(table['cells']):
        return True
    else:
        return False


def tensor_to_image(tensor):
    image = tensor.detach().cpu().numpy()
    if (len(image.shape) == 3) and (image.shape[0] != 3) and (image.shape[0] != 1):
        image = np.sqrt(np.sum(np.power(image, 2), axis=0, keepdims=True))
    image = 255 * (image-np.min(image))/(np.max(image) - np.min(image))
    image = image.astype(np.uint8)
    if len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0)).copy()
        if image.shape[2] == 1:
            image = image[:, :, 0]
    return image


def visualize_layout(image, table):
    def draw_segmentation(image, segmentation, color):
        for contour in segmentation:
            contour = np.array(contour, dtype=np.int32)
            image = cv2.polylines(image, [contour], True, color)
        return image
    for cell in table['cells']:
        if 'segmentation' in cell:
            image = draw_segmentation(image, cell['segmentation'], (255, 0, 0))
    return image

virtual_chars = ["<b>", "</b>", "<i>", "</i>", "<sup>", "</sup>", "<sub>", "</sub>", "<overline>", "</overline>", "<underline>", "</underline>", "<strike>", "</strike>"]


def is_blank(content):
    global virtual_chars
    
    new_content = content
    for item in virtual_chars:
        new_content = new_content.replace(item, '')
    return new_content.strip() == ''


def filt_content(content, filt_blank=False, filt_virtual=False, filt_pad=False):
    global virtual_chars
    if filt_blank:
        if is_blank(content):
            content = ''

    if filt_virtual:
        for item in content:
            content = content.replace(item, '')

    if filt_pad:
        content = content.strip()

    return content


def filt_transcript(html, filt_blank=False, filt_virtual=False, filt_pad=False):
    start_idx = 0
    while '<td' in html[start_idx:]:
        start_idx = html[start_idx:].index('<td') + start_idx
        content_start_idx = html[start_idx:].index('>') + 1 + start_idx
        content_end_idx = html[content_start_idx:].index('</td>') + content_start_idx
        end_idx = content_end_idx + len('</td>')

        content = html[content_start_idx:content_end_idx]
        content = filt_content(content, filt_blank, filt_virtual, filt_pad)
        html = html[:content_start_idx] + content + html[content_end_idx:]
        start_idx = end_idx - (content_end_idx-content_start_idx - len(content))
    return html
