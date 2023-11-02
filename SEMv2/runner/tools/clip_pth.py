import torch
from collections import OrderedDict


model_path = '/work1/cv1/jszhang6/TSR/code/SEMv2/00_Spliter/experiments/default/latest_model.pth'
save_path = '/work1/cv1/jszhang6/TSR/code/SEMv2/00_Spliter/experiments/default/pretrained_detection.pth'

state_dict = torch.load(model_path, map_location="cpu")['model_param']
detection_dict = dict()
detection_dict['model_param'] = OrderedDict()
for name in state_dict:
    if name.startswith(tuple(['backbone','neck', 'row_bbox_head', 'col_bbox_head'])):
        detection_dict['model_param'][name] = state_dict[name]

torch.save(detection_dict, save_path)