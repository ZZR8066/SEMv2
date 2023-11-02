import cv2
import mmcv
import numpy as np
from .utils import Resize, PhotoMetricDistortion
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data


class CallResizeImage:
    def __init__(self, **kwargs):
        self.operation = Resize(**kwargs)
    
    def __call__(self, image, table):
        table.update(img=image)
        table = self.operation(table)
        image = table['img']
        return image, table


class CallImageDistortion:
    def __init__(self, **kwargs):
        self.operation = PhotoMetricDistortion(**kwargs)
    
    def __call__(self, image, table):
        image = self.operation(image).astype('uint8')
        return image, table


class CallRowColStartBox:
    def __call__(self, image, table):
        table.update(img=image)
        
        if 'row_start_center_bboxes' in table and 'col_start_center_bboxes' in table:
            row_start_bboxes = np.array(table['row_start_center_bboxes'], dtype=np.float32)
            col_start_bboxes = np.array(table['col_start_center_bboxes'], dtype=np.float32)
        else:
            row_start_bboxes = np.zeros((0, 4), dtype=np.float32)
            col_start_bboxes = np.zeros((0, 4), dtype=np.float32)
        
        return image, table, row_start_bboxes, col_start_bboxes


class CallImageNormalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image, table, row_start_bboxes, col_start_bboxes):
        image = mmcv.impad_to_multiple(image, divisor=32, pad_val=0)
        image = F.to_tensor(image)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image, table, row_start_bboxes, col_start_bboxes


class CallRowColLineMask:
    def __call__(self, image, table, row_start_bboxes, col_start_bboxes):
        if 'row_line_segmentations' in table and  'col_line_segmentations' in table:
            image_h, image_w = image.shape[-2:]

            row_line_masks = list()
            for segm in table['row_line_segmentations']:
                canvas = np.zeros((image_h, image_w))
                segm = np.array(segm).reshape(-1,1,2).astype('int')
                cv2.polylines(canvas, [segm], False, 1, 5)
                row_line_masks.append(canvas)
            row_line_masks = np.concatenate([item[None] for item in row_line_masks], axis=0).astype(np.int64)

            col_line_masks = list()
            for segm in table['col_line_segmentations']:
                canvas = np.zeros((image_h, image_w))
                segm = np.array(segm).reshape(-1,1,2).astype('int')
                cv2.polylines(canvas, [segm], False, 1, 5)
                col_line_masks.append(canvas)
            col_line_masks = np.concatenate([item[None] for item in col_line_masks], axis=0).astype(np.int64)

        else:
            image_h, image_w = image.shape[-2:] # (3, h, w)
            row_line_masks = np.zeros((0, image_h, image_w), dtype=np.int64)
            col_line_masks = np.zeros((0, image_h, image_w), dtype=np.int64)
        return image, table, row_start_bboxes, col_start_bboxes, row_line_masks, col_line_masks


class CallLayout:
    def __call__(self, image, table, row_start_bboxes, col_start_bboxes, row_line_masks, col_line_masks):
        if 'layout' in table:
            layout = np.array(table['layout'], dtype=np.int64) # (NumRow, NumCol)
        else:
            layout = np.zeros((0, 0),  dtype=np.int64)
        return image, table, row_start_bboxes, col_start_bboxes, row_line_masks, col_line_masks, layout