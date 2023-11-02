import json
import numpy as np
from utils.utils import extend_text_lines
from utils.cal_f1 import table_to_relations, evaluate_f1
from utils.metric import TEDSMetric
from utils.format_translate import table_to_html, format_html

label_path = '/work1/cv1/jszhang6/TSR/datasets/companydataset/output_dir/csig_challenge/evaluation/00000-gt.json'
pred_path = '/work1/cv1/jszhang6/TSR/datasets/companydataset/output_dir/csig_challenge/evaluation/00000-pred.json'
json_path = '/work1/cv1/jszhang6/TSR/datasets/companydataset/output_dir/csig_challenge/valid/00000.json' # do not provide to competitors

label = json.load(open(label_path, 'r'))
pred = json.load(open(pred_path, 'r'))
info = json.load(open(json_path, 'r'))

# extend text line to predicted result
pred['cells'] = extend_text_lines(pred['cells'], info['line'])

# trans layout to np.narray
pred['layout'] = np.array(pred['layout'])
label['layout'] = np.array(label['layout'])

# calculate F1-Measure
pred_relations = table_to_relations(pred)
label_relations = table_to_relations(label)
f1 = evaluate_f1([label_relations], [pred_relations], num_workers=1)

# calculate TEDS-Struct
pred_htmls = table_to_html(pred)
pred_htmls = format_html(pred_htmls)

label_htmls = table_to_html(label)
label_htmls = format_html(label_htmls)

teds_metric = TEDSMetric(num_workers=1, structure_only=False)
teds_info = teds_metric([pred_htmls], [label_htmls])

# calculate final metric base on macro
metric = 0
for idx in range(len(teds_info)):
    metric += 0.5 * f1[idx][-1] + 0.5 * teds_info[idx]
metric = metric / len(teds_info) * 100
print('final metric is %.2f' % metric)