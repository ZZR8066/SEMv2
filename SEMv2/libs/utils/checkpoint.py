import os
import torch
from .comm import get_rank, synchronize


def save_checkpoint(checkpoint, model, optimizer=None, best_metric=None, epoch=None):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if get_rank() == 0:
        if not os.path.exists(os.path.dirname(checkpoint)):
            os.makedirs(os.path.dirname(checkpoint))
        
        infos = dict()
        infos['model_param'] = model.state_dict()
        if optimizer is not None:
            infos['opt_param'] = optimizer.state_dict()
        
        if best_metric is not None:
            infos['best_metric'] = best_metric
        
        if epoch is not None:
            infos['epoch'] = epoch

        torch.save(infos, checkpoint)
    synchronize()


def load_checkpoint(checkpoint, model, optimizer=None):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    checkpoint = torch.load(checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model_param'], strict=False)

    if (optimizer is not None) and ('opt_param' in checkpoint):
        optimizer.load_state_dict(checkpoint['opt_param'])

    if 'best_metric' in checkpoint:
        best_metric = checkpoint['best_metric']
    else:
        best_metric = None
    
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
    else:
        epoch = None
    return best_metric, epoch