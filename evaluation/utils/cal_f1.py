import tqdm
import copy
import Polygon
import numpy as np
from .scitsr.eval import json2Relations, eval_relations


def cal_cell_spans(table):
    layout = table['layout']
    num_cells = len(table['cells'])
    cells_span = list()
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout == cell_id)
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        # assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return cells_span


def table_to_relations(table):
    relations = []
    for cell in table['cells']:
        x1, y1, x2, y2 = cell['col_start_idx'], cell['row_start_idx'], cell['col_end_idx'], cell['row_end_idx'] 
        content = cell['transcript']
        relations.append(dict(start_row=y1, end_row=y2, start_col=x1, end_col=x2, content=content))
    return dict(cells=relations)


def cal_f1(label, pred):
    label = json2Relations(label, splitted_content=True)
    pred = json2Relations(pred, splitted_content=True)
    precision, recall = eval_relations(gt=[label], res=[pred], cmp_blank=True)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return [precision, recall, f1]


def single_process(labels, preds):
    scores = dict()
    for key in (labels.keys()):
        pred = preds.get(key, '')
        label = labels.get(key, '')
        score = cal_f1(label, pred)
        scores[key] = score
    return scores


def _worker(labels, preds,  keys, result_queue):
    for key in keys:
        label = labels.get(key, '')
        pred = preds.get(key, '')
        score = cal_f1(label, pred)
        result_queue.put((key, score))


def multi_process(labels, preds, num_workers):
    import multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    keys = list(labels.keys())
    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker,
            args=(
                labels,
                preds,
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
        P, R, F1 = (100 * np.array(list(scores.values()))).mean(0).tolist()
        tq.set_description('P: %.2f, R: %.2f, F1: %.2f' % (P, R, F1), False)
        tq.update()
    
    return scores


def evaluate_f1(labels, preds, num_workers=0):
    preds = {idx: pred for idx, pred in enumerate(preds)}
    labels = {idx: label for idx, label in enumerate(labels)}
    if num_workers <= 1:
        scores = single_process(labels, preds)
    else:
        scores = multi_process(labels, preds, num_workers)
    sorted_idx = sorted(list(range(len(list(scores)))), key=lambda idx: list(scores.keys())[idx])
    scores = [scores[idx] for idx in sorted_idx]
    return scores