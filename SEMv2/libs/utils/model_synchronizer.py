import torch
from .comm import get_world_size
import torch.distributed as dist


class ModelSynchronizer:
    bm_map = {
        2: 0.65,
        4: 0.75,
        8: 0.875,
        12: 0.8875,
        16: 0.9,
        32: 0.9
    }

    def __init__(self, model, sync_rate, bm=None, blr=1.0, rescale_grad=1.0):
        if bm is None:
            self.bm = self.bm_map[get_world_size()]
        else:
            self.bm = bm
        self.blr = blr
        self.model = model
        self.sync_rate = sync_rate
        self.rescale_grad = rescale_grad
        self.count = 0

        self.param_align()

        self.momentums = dict()
        self.global_params = dict()
        for k, v in self.model.named_parameters():
            temp = torch.zeros_like(v, requires_grad=False)
            temp.copy_(v.data)
            self.global_params[k] = v
            self.momentums[k] = torch.zeros_like(v, requires_grad=False)
    
    def param_align(self):
        for v in self.model.parameters():
            dist.broadcast_multigpu([v.data], src=0)

        for k, v in self.model.named_buffers():
            if 'num_batches_tracked' in k:
                continue
            dist.broadcast_multigpu([v.data], src=0)

    def sync_params(self):
        size = float(get_world_size())
        for v in self.model.parameters():
            dist.all_reduce(v.data, op=dist.ReduceOp.SUM)
            v.data /= size

        for k, v in self.model.named_buffers():
            if 'num_batches_tracked' in k:
                continue
            dist.all_reduce(v.data, op=dist.ReduceOp.SUM)
            v.data /= size

    def __call__(self, final_align=False):
        self.count += 1
        if (self.count % self.sync_rate == 0) or final_align:
            with torch.no_grad():
                if final_align:
                    self.param_align()
                else:
                    self.sync_params()

                    for k, v in self.model.named_parameters():
                        global_param = self.global_params[k]
                        momentum = self.momentums[k]
                        grad = v.data * self.rescale_grad - global_param
                        momentum *= self.bm
                        global_param -= momentum
                        momentum += self.blr * grad
                        global_param += (1.0 + self.bm) * momentum
                        v.detach().copy_(global_param.detach())
