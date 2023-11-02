import cv2
import torch
import numpy as np
from .list_record_cache import ListRecordLoader


class LRCRecordLoader:
    def __init__(self, lrc_path):
        self.loader = ListRecordLoader(lrc_path)

    def __len__(self):
        return len(self.loader)
    
    def get_info(self, idx):
        table = self.loader.get_record(idx)
        w = table['image_w']
        h = table['image_h']
        n_rows, n_cols = table['layout'].shape
        n_cells = n_rows * n_cols
        return w, h, n_cells

    def get_data(self, idx):
        table = self.loader.get_record(idx)
        image = cv2.imread(table['image_path'].replace('/work1/', '/work2/'))
        return image, table


class Dataset:
    def __init__(self, loaders, transforms):
        self.loaders = loaders
        self.transforms = transforms

    def _match_loader(self, idx):
        offset = 0
        for loader in self.loaders:
            if len(loader) + offset > idx:
                return loader, idx - offset
            else:
                offset += len(loader)
        raise IndexError()

    def get_info(self, idx):
        loader, rela_idx = self._match_loader(idx)
        return loader.get_info(rela_idx)

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def __getitem__(self, idx):
        try:
            loader, rela_idx = self._match_loader(idx)
            image, table = loader.get_data(rela_idx)
            image, table, row_start_bboxes, col_start_bboxes, \
                row_line_masks, col_line_masks, layout = self.transforms(image, table)
            image_h, image_w = table['image_h'], table['image_w'] # the resize image shape
            # (image.shape[2], image.shape[1]) --> pad shape
            image_size = np.array([image_w, image_h], dtype=np.int64) # origin shape
            return dict(
                image_size=image_size,
                image=image,
                row_start_bboxes=row_start_bboxes,
                col_start_bboxes=col_start_bboxes,
                row_line_masks=row_line_masks,
                col_line_masks=col_line_masks,
                layout=layout,
                table=table
            )
        except Exception as e:
            print('Error occured while load data: %d' % idx)
            raise e


def collate_func(batch_data):
    def merge1d(tensors, pad_id):
        lengths= [len(s) for s in tensors]
        out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
        for i, s in enumerate(tensors):
            out[i,:len(s)] = s
        return out

    def merge2d(tensors, pad_id):
        dim1 = max([s.shape[0] for s in tensors])
        dim2 = max([s.shape[1] for s in tensors])
        out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
        for i, s in enumerate(tensors):
            out[i, :s.shape[0], :s.shape[1]] = s
        return out

    def merge3d(tensors, pad_id):
        dim1 = max([s.shape[0] for s in tensors])
        dim2 = max([s.shape[1] for s in tensors])
        dim3 = max([s.shape[2] for s in tensors])
        out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
        for i, s in enumerate(tensors):
            out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
        return out

    images = merge3d([data['image'] for data in batch_data], 0) # (B, 3, H, W)
    images_size = merge1d([torch.from_numpy(data['image_size']) for data in batch_data], 0) # (B, 2) -> [(H1,W1), ..., (Hn,Wn)]

    row_start_bboxes = [torch.from_numpy(data['row_start_bboxes']) for data in batch_data] # [(NumRow, 4)] -> [[(x1,y1,x2,y2)]]
    row_line_masks = merge3d([torch.from_numpy(data['row_line_masks']) for data in batch_data], 0) # [(NumRow, H, W)]

    col_start_bboxes = [torch.from_numpy(data['col_start_bboxes']) for data in batch_data] # [NumCol, 4)] -> [[(x1,y1,x2,y2)]]
    col_line_masks = merge3d([torch.from_numpy(data['col_line_masks']) for data in batch_data], 0) # [(NumCol, H, W)]
    
    layouts = merge2d([torch.from_numpy(data['layout']) for data in batch_data], -100) # [(NumRow, NumCol)]
    tables = [data['table'] for data in batch_data]

    return dict(
        images=images,
        images_size=images_size,
        row_start_bboxes=row_start_bboxes,
        row_line_masks=row_line_masks,
        col_start_bboxes=col_start_bboxes,
        col_line_masks=col_line_masks,
        layouts=layouts,
        tables=tables
    )
