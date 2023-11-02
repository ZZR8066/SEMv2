import re
import copy
import Polygon
import numpy as np
from bs4 import BeautifulSoup as bs
from .time_counter import format_table


def check_continuous(seq):
    if len(seq) > 0:
        pre_val = seq[0]
        for val in seq[1:]:
            assert pre_val + 1 == val
            pre_val = val

def table_to_latex(table):
    def cal_cls_id(transcript):
        transcript = ''.join(transcript)
        if transcript == '':
            return '</none>'
        elif transcript == '<b> </b>':
            return '</bold>'
        elif transcript == ' ':
            return '</space>'
        else:
            return '</line>'
    assert table['layout'].max() + 1 == len(table['cells'])
    latex = [cal_cls_id(cell['transcript']) for cell in table['cells']]
    return latex

def html_to_table(html):
    tokens = html['html']['structure']['tokens']

    layout = [[]]

    def extend_table(x, y):
        assert (x >= 0) and (y >= 0)
        nonlocal layout

        if x >= len(layout[0]):
            for row in layout:
                row.extend([-1] * (x - len(row) + 1))
        
        if y >= len(layout):
            for _ in range(y - len(layout) + 1):
                layout.append([-1] * len(layout[0]))

    def set_cell_val(x, y, val):
        assert (x >= 0) and (y >= 0)
        nonlocal layout
        extend_table(x, y)
        layout[y][x] = val

    def get_cell_val(x, y):
        assert (x >= 0) and (y >= 0)
        nonlocal layout
        extend_table(x, y)
        return layout[y][x]

    def parse_span_val(token):
        span_val = int(token[token.index('"') + 1:token.rindex('"')])
        return span_val

    def maskout_left_rows():
        nonlocal row_idx, layout
        layout = layout[:max(row_idx+1, 1)]

    row_idx = -1
    col_idx = -1
    line_idx = -1
    inside_head = False
    inside_body = False
    head_rows = list()
    body_rows = list()
    col_span = 1
    row_span = 1
    for token in tokens:
        if token == '<thead>':
            inside_head = True
            maskout_left_rows()
        elif token == '</thead>':
            inside_head = False
            maskout_left_rows()
        elif token == '<tbody>':
            inside_body = True
            maskout_left_rows()
        elif token == '</tbody>':
            inside_body = False
            maskout_left_rows()
        elif token == '<tr>':
            row_idx += 1
            col_idx = -1
            if inside_head:
                head_rows.append(row_idx)
            if inside_body:
                body_rows.append(row_idx)
        elif token in ['<td>', '<td']:
            line_idx += 1
            col_idx += 1
            row_span = 1
            col_span = 1
            while get_cell_val(col_idx, row_idx) != -1:
                col_idx += 1
        elif 'colspan' in token:
            col_span = parse_span_val(token)
        elif 'rowspan' in token:
            row_span = parse_span_val(token)
        elif token == '</td>':
            for cur_row_idx in range(row_idx, row_idx + row_span):
                for cur_col_idx in range(col_idx, col_idx + col_span):
                    set_cell_val(cur_col_idx, cur_row_idx, line_idx)
            col_idx += col_span - 1

    check_continuous(head_rows)
    check_continuous(body_rows)
    assert len(set(head_rows) | set(body_rows)) == len(layout)
    layout = np.array(layout)
    assert np.all(layout >= 0)

    cells_info = list()
    for cell_idx, cell in enumerate(html['html']['cells']):
        transcript = cell['tokens']
        cell_info = dict(
            transcript=transcript
        )
        if 'bbox' in cell:
            x1, y1, x2, y2 = cell['bbox']
            cell_info['bbox'] = [x1, y1, x2, y2]
            cell_info['segmentation'] = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        cells_info.append(cell_info)
    
    table = dict(
        layout=layout,
        cells=cells_info,
        head_rows=head_rows,
        body_rows=body_rows
    )
    return table


def segmentation_to_bbox(segmentation):
    x1 = min([min([pt[0] for pt in contour]) for contour in segmentation])
    y1 = min([min([pt[1] for pt in contour]) for contour in segmentation])
    x2 = max([max([pt[0] for pt in contour]) for contour in segmentation])
    y2 = max([max([pt[1] for pt in contour]) for contour in segmentation])
    return [x1, y1, x2, y2]


def table_to_html(table):
    layout = table['layout']
   
    # head_rows = table['head_rows']
    # body_rows = table['body_rows']

    head_rows = []
    body_rows = [i for i in range(layout.shape[0])]

    cells_span = list()
    for cell_idx in range(len(table['cells'])):
        cell_positions = np.argwhere(layout == cell_idx)
        row_span = [np.min(cell_positions[:, 0]), np.max(cell_positions[:, 0]) + 1]
        col_span = [np.min(cell_positions[:, 1]), np.max(cell_positions[:, 1]) + 1]
        
        # assert np.all(layout[row_span[0]:row_span[1], col_span[0]:col_span[1]] == cell_idx)
      
        cells_span.append([row_span, col_span])

    cells = list()
    tokens = ['<thead>']
    inside_head = True
    for row_idx in range(layout.shape[0]):
        if row_idx in body_rows:
            if inside_head:
                tokens.append('</thead>')
                tokens.append('<tbody>')
                inside_head = False
        tokens.append('<tr>')
        for col_idx in range(table['layout'].shape[1]):
            cell_idx = layout[row_idx][col_idx]
            assert cell_idx <= len(cells)
            if cell_idx == len(cells):
                row_span, col_span = cells_span[cell_idx]
                if (row_span[1] - row_span[0]) == 1 and (col_span[1] - col_span[0] == 1):
                    tokens.append('<td>')
                else:
                    tokens.append('<td')
                    if (row_span[1] - row_span[0]) > 1:
                        tokens.append(' rowspan="%d"' % (row_span[1] - row_span[0]))
                    if (col_span[1] - col_span[0]) > 1:
                        tokens.append(' colspan="%d"' % (col_span[1] - col_span[0]))
                    tokens.append('>')
                tokens.append('</td>')
                
                cell = dict()
                cell['tokens'] = table['cells'][cell_idx]['transcript']
                if 'segmentation' in table['cells'][cell_idx]:
                    cell['bbox'] = segmentation_to_bbox(table['cells'][cell_idx]['segmentation'])
                cells.append(cell)
        tokens.append('</tr>')
    if inside_head:
        tokens.append('</thead>')
        tokens.append('<tbody>')
    tokens.append('</tbody>')
   
    html = dict(
        html=dict(
            cells=cells,
            structure=dict(
                tokens=tokens
            )
        )
    )
    return html


def format_html_for_vis(html):
    html_string = '''<html>
                     <head>
                     <meta charset="UTF-8">
                     <style>
                     table, th, td {
                       border: 1px solid black;
                       font-size: 10px;
                     }
                     </style>
                     </head>
                     <body>
                     <table frame="hsides" rules="groups" width="100%%">
                         %s
                     </table>
                     </body>
                     </html>''' % ''.join(html['html']['structure']['tokens'])
    cell_nodes = list(re.finditer(r'(<td[^<>]*>)(</td>)', html_string))
    assert len(cell_nodes) == len(html['html']['cells']), 'Number of cells defined in tags does not match the length of cells'
    cells = [''.join(c['tokens']) for c in html['html']['cells']]
    offset = 0
    for n, cell in zip(cell_nodes, cells):
        html_string = html_string[:n.end(1) + offset] + cell + html_string[n.start(2) + offset:]
        offset += len(cell)
    # prettify the html
    soup = bs(html_string)
    html_string = soup.prettify()
    return html_string


def format_html(html):
    html_string = '''<html><body><table>%s</table></body></html>''' % ''.join(html['html']['structure']['tokens'])
    cell_nodes = list(re.finditer(r'(<td[^<>]*>)(</td>)', html_string))
    assert len(cell_nodes) == len(html['html']['cells']), 'Number of cells defined in tags does not match the length of cells'
    cells = [''.join(c['tokens']) for c in html['html']['cells']]
    offset = 0
    for n, cell in zip(cell_nodes, cells):
        html_string = html_string[:n.end(1) + offset] + cell + html_string[n.start(2) + offset:]
        offset += len(cell)
    return html_string


def format_table_layout(table):
    layout = table['table']['layout']
    cell_lines = [cell['lines_idx'] for cell in table['table']['cells']]

    table_cells_info = list()
    for row in layout:
        row_cells_info = list()
        for cell_idx in row:
            cell_str = ','.join([str(item) for item in cell_lines[cell_idx]])
            row_cells_info.append(cell_str)
        table_cells_info.append(row_cells_info)
    
    return format_table(table_cells_info, padding=1)


def remove_blank_cell(html):
    start_idx = 0
    while '<td' in html[start_idx:]:
        start_idx = html[start_idx:].index('<td') + start_idx
        content_start_idx = html[start_idx:].index('>') + 1 + start_idx
        content_end_idx = html[content_start_idx:].index('</td>') + content_start_idx
        end_idx = content_end_idx + len('</td>')
        if content_end_idx == content_start_idx:
            html = html[:start_idx] + html[end_idx:]
        else:
            start_idx = end_idx
    return html
