import torch
from torch.utils.data.distributed import DistributedSampler

from . import transform as T
from .dataset import LRCRecordLoader
from .batch_sampler import BucketSampler
from .dataset import Dataset, collate_func
from libs.utils.comm import distributed, get_rank, get_world_size


def create_train_dataloader(lrcs_path, num_workers, max_batch_size, max_pixel_nums, bucket_seps):
    loaders = list()
    for lrc_path in lrcs_path:
        loader = LRCRecordLoader(lrc_path)
        loaders.append(loader)

    transforms = T.Compose([
        T.CallResizeImage(ratio_range=(0.7,1.2), keep_ratio=True),
        T.CallImageDistortion(brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
        T.CallRowColStartBox(),
        T.CallImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.CallRowColLineMask(),
        T.CallLayout()
    ])
    
    dataset = Dataset(loaders, transforms)
    batch_sampler = BucketSampler(dataset, get_world_size(), get_rank(), max_pixel_nums=max_pixel_nums, max_batch_size=max_batch_size, seps=bucket_seps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_func,
        batch_sampler=batch_sampler
    )
    return dataloader


def create_valid_dataloader(lrc_path, num_workers, batch_size):
    loader = LRCRecordLoader(lrc_path)

    transforms = T.Compose([
        T.CallResizeImage(ratio_range=(0.99,1.0), keep_ratio=True),
        T.CallRowColStartBox(),
        T.CallImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.CallRowColLineMask(),
        T.CallLayout()
    ])
    
    dataset = Dataset([loader], transforms)
    if distributed():
        sampler = DistributedSampler(dataset, get_world_size(), get_rank(), True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_func,
            sampler=sampler,
            drop_last=False
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_func,
            shuffle=False,
            drop_last=False
        )
    return dataloader