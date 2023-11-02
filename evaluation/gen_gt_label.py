# generate ground-true label for evaluation
import json
from utils.utils import parse_relation_from_table, get_span_cells, get_shared_line, get_shared_line_id, sort_shared_line, parse_gt_label, update_gt_label

image_path = '/work1/cv1/jszhang6/TSR/datasets/companydataset/output_dir/csig_challenge/valid/00000.png'
json_path = '/work1/cv1/jszhang6/TSR/datasets/companydataset/output_dir/csig_challenge/valid/00000.json'
output_path = '/work1/cv1/jszhang6/TSR/datasets/companydataset/output_dir/csig_challenge/evaluation/00000-gt.json'

table = json.load(open(json_path, 'r'))
table = parse_relation_from_table(table)
span_indice, row_span_indice, col_span_indice = get_span_cells(table['row_adj'], table['col_adj'])
shared_row_lines = get_shared_line(table['row_adj'], table['cell_adj'], table, row_span_indice)
shared_col_lines = get_shared_line(table['col_adj'], table['cell_adj'], table, col_span_indice)
shared_row_line_ids = get_shared_line_id(table['row_adj'], table['cell_adj'], row_span_indice)
shared_col_line_ids = get_shared_line_id(table['col_adj'], table['cell_adj'], col_span_indice)
shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines = \
        sort_shared_line(shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines)
gt_label = parse_gt_label(table['cell_adj'], table['row_adj'], table['col_adj'], shared_row_line_ids, shared_col_line_ids)
gt_label = update_gt_label(gt_label, table) # update transcripts for wired table
json.dump(gt_label, open(output_path, 'w'), indent=4)
