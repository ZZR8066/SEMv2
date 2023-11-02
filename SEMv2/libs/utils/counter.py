import torch
from collections import defaultdict
from .comm import distributed, all_gather


def format_dict(res_dict):
    res_strs = []
    for key, val in res_dict.items():
        res_strs.append('%s: %s' % (key, val))
    return ', '.join(res_strs)


class Counter:
    def __init__(self, cache_nums=1000):
        self.cache_nums = cache_nums
        self.reset()

    def update(self, metric):
        for key, val in metric.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.metric_dict[key].append(val)
            if self.cache_nums is not None:
                self.metric_dict[key] = self.metric_dict[key][-1*self.cache_nums:]

    def reset(self):
        self.metric_dict = defaultdict(list)

    def _sync(self):
        metric_dicts = all_gather(self.metric_dict)
        total_metric_dict = defaultdict(list)
        for metric_dict in metric_dicts:
            for key, val in metric_dict.items():
                total_metric_dict[key].extend(val)
        return total_metric_dict

    def format_mean(self, sync=True):
        if sync and distributed():
            metric_dict = self._sync()
        else:
            metric_dict = self.metric_dict
        res_dict = {key: '%.4f' % (sum(val)/len(val)) for key, val in metric_dict.items()}
        return format_dict(res_dict)
