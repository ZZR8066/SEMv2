import sys
import json
sys.path.append('./')
sys.path.append('../')
import os
import tqdm
import torch
import numpy as np
from collections import defaultdict
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.utils import is_simple_table
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1
from libs.utils.format_translate import table_to_html, format_html
from libs.utils.metric import  TEDSMetric
from libs.utils.checkpoint import load_checkpoint
from libs.utils.comm import synchronize, all_gather


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrc", type=str, default=None)
    parser.add_argument("--cfg", type=str, default='default')
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    setup_config(args.cfg)
    if args.lrc is not None:
        cfg.valid_lrc_path = args.lrc
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('Line Detect Model', cfg.work_dir, 'valid.log')
    logger.info('Use config: %s' % args.cfg)
    logger.info('Evaluate Dataset: %s' % cfg.valid_lrc_path)


def valid(cfg, dataloader, model):
    model.eval()
    total_label_relations = list()
    total_pred_relations = list()
    total_relations_metric = list()
    total_label_htmls = list()
    total_pred_htmls = list()
    total_tables_type = list()
    
    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        images = data_batch['images'].to(cfg.device)
        images_size = data_batch['images_size'].to(cfg.device)
        tables = data_batch['tables']
        
        # pred
        _, row_start_bboxes, row_center_points, row_segm_logits, \
            col_start_bboxes, col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols = model(images, images_size)

        pred_tables = [
            pred_result_to_table(tables[batch_idx],
                (row_center_points[batch_idx], row_segm_logits[batch_idx], \
                    col_center_points[batch_idx], col_segm_logits[batch_idx], \
                        mg_logits[batch_idx], num_rows[batch_idx], num_cols[batch_idx])
            ) \
            for batch_idx in range(len(images_size))
        ]
        pred_relations = [table_to_relations(table) for table in pred_tables]
        total_pred_relations.extend(pred_relations)

        label_relations = [table_to_relations(table) for table in tables]
        total_label_relations.extend(label_relations)

        pred_htmls = [table_to_html(table) for table in pred_tables]
        total_pred_htmls.extend([format_html(item) for item in pred_htmls])

        label_htmls = [table_to_html(table) for table in tables]
        total_label_htmls.extend([format_html(item) for item in label_htmls])

        tables_type = ['Simple' if is_simple_table(table) else 'Complex' for table in tables]
        total_tables_type.extend(tables_type)

    # new evaluate P, R, F1, Acc
    total_relations_metric = evaluate_f1(total_label_relations, total_pred_relations, num_workers=40)
    total_relations_metric_p  = [sum(np.array(total_relations_metric)[:, 0].tolist())]
    total_relations_metric_r  = [sum(np.array(total_relations_metric)[:, 1].tolist())]
    total_len = [float(len(total_relations_metric))]

    total_relations_metric_p = all_gather(total_relations_metric_p)
    total_relations_metric_r = all_gather(total_relations_metric_r)
    total_len = all_gather(total_len)
    P = np.array(total_relations_metric_p).sum() / np.array(total_len).sum()
    R = np.array(total_relations_metric_r).sum() / np.array(total_len).sum()
    F1 = 2 * P * R / (P + R)
    logger.info('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))

    # evaluate TEDS
    teds_metric = TEDSMetric(num_workers=40, structure_only=False)
    teds_info = teds_metric(total_pred_htmls, total_label_htmls)
    
    teds = (sum(teds_info), len(teds_info))
    teds = [sum(item) for item in zip(*all_gather(teds))]
    teds = teds[0]/teds[1]

    total_correct = [(np.array(teds_info)==1).sum()]
    total_correct = all_gather(total_correct)
    Acc = np.array(total_correct).sum() / np.array(total_len).sum()

    typed_teds = defaultdict(list)
    for table_type, teds_info_ps in zip(total_tables_type, teds_info):
        typed_teds[table_type].append(teds_info_ps)
    typed_teds = {key: (sum(val), len(val)) for key, val in typed_teds.items()}

    total_typed_teds = dict()
    for typed_teds_pw in all_gather(typed_teds):
        for key, val in typed_teds_pw.items():
            if key not in total_typed_teds:
                total_typed_teds[key] = val
            else:
                exist_val = total_typed_teds[key]
                total_typed_teds[key] = (val[0] + exist_val[0], val[1] + exist_val[1])
    typed_teds = {key: val[0]/val[1] for key, val in total_typed_teds.items()}

    logger.info('[Valid] Teds: %s, Acc: %s' % (teds, Acc))
    for key, val in typed_teds.items():
        logger.info('    %s Teds: %s' % (key, val))
    
    return (teds, )


def main():
    init()

    valid_dataloader = create_valid_dataloader(
        cfg.valid_lrc_path,
        cfg.valid_num_workers,
        cfg.valid_batch_size
    )
    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(valid_dataloader.dataset),
                len(valid_dataloader.batch_sampler),
                valid_dataloader.batch_size
            )
    )

    model = build_model(cfg)
    model.cuda()
    
    # eval_checkpoint = os.path.join(cfg.work_dir, 'best_f1_model_c1.pth')
    # eval_checkpoint = os.path.join(cfg.work_dir, 'best_f1_model_c2.pth')
    # eval_checkpoint = os.path.join(cfg.work_dir, 'best_f1_model_c3.pth')
    # eval_checkpoint = os.path.join(cfg.work_dir, 'best_f1_model_c5.pth')
    eval_checkpoint = os.path.join(cfg.work_dir, 'best_f1_model.pth')
    load_checkpoint(eval_checkpoint, model)
    logger.info('Load checkpoint from: %s' % eval_checkpoint)

    with torch.no_grad():
        valid(cfg, valid_dataloader, model)


if __name__ == '__main__':
    main()