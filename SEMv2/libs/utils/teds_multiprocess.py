import json
import tqdm
from libs.utils.teds import TEDS
from collections import defaultdict


# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('pred_path', type=str, default=None)
#     parser.add_argument('label_path', type=str, default=None)
#     parser.add_argument('-s', '--structure_only', action='store_true')
#     parser.add_argument('-n', '--num_workers', type=int, default=1)
#     args = parser.parse_args()
#     return args


def is_simple(data):
    if ('colspan' in data) or ('rowspan' in data):
        return False
    else:
        return True


def judge_type(data):
    if is_simple(data):
        return 'Simple'
    else:
        return 'Complex'


def single_process(pred_htmls, label_htmls, structure_only=False):
    evaluator = TEDS(structure_only=structure_only)
    scores = dict()
    for key in tqdm.tqdm(label_htmls.keys()):
        pred_html = pred_htmls.get(key, '')
        label_html = label_htmls[key]['html']
        score = evaluator.evaluate(pred_html, label_html)
        scores[key] = score
    return scores


def _worker(pred_htmls, label_htmls, keys, result_queue, structure_only=False):
    evaluator = TEDS(structure_only=structure_only)
    for key in keys:
        pred_html = pred_htmls.get(key, '')
        label_html = label_htmls[key]['html']
        score = evaluator.evaluate(pred_html, label_html)
        result_queue.put((key, score))


def multi_process(pred_htmls, label_htmls, num_workers, structure_only=False):
    import multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    keys = list(label_htmls.keys())
    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker,
            args=(
                pred_htmls,
                label_htmls,
                keys[worker_idx::num_workers],
                result_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)
    scores = dict()
    tq = tqdm.tqdm(total=len(keys))
    for _ in range(len(keys)):
        key, val = result_queue.get()
        scores[key] = val
        teds = sum(scores.values()) / len(scores)
        tq.set_description('Teds: %s' % teds, False)
        tq.update()
    tq.close()
    return scores


def evaluate(pred_htmls, label_htmls, num_workers, structure_only=False):
    if num_workers <= 1:
        scores = single_process(pred_htmls, label_htmls, structure_only)
    else:
        scores = multi_process(pred_htmls, label_htmls, num_workers, structure_only)
    teds = sum(scores.values())/len(scores)

    typed_teds = defaultdict(list)
    for key, score in scores.items():
        data_type = judge_type(label_htmls[key]['html'])
        typed_teds[data_type].append(score)
    
    typed_teds = {key: sum(val)/len(val) for key, val in typed_teds.items()}
    return teds, typed_teds


# def main():
#     args = parse_args()
#     pred_data = json.load(open(args.pred_path))
#     label_data = json.load(open(args.label_path))
    
#     teds, typed_teds = evaluate(pred_data, label_data, args.num_workers, args.structure_only)
#     print('Teds: %s' % teds)
#     for key, val in typed_teds.items():
#         print('    %s Teds: %s' % (key, val))


# if __name__ == '__main__':
#     main()
