import sys
import json
sys.path.append('./')
sys.path.append('../')
import os
import tqdm
import torch
import numpy as np
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1, cal_f1
from libs.utils.visualize import visualize
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
    
    vis_dir = os.path.join(cfg.work_dir, 'vis_folder')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    vis_dir = os.path.abspath(vis_dir)

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
        label_relations = [table_to_relations(table) for table in tables]

        for batch_idx in range(len(tables)):
            relations_metric = cal_f1(label_relations[batch_idx], pred_relations[batch_idx])[-1]
            if relations_metric == 1:
                continue
            prefix = os.path.join(os.path.abspath(vis_dir), 'f1-%.2f-id-%04d' % (relations_metric*100, it * len(tables) + batch_idx))
            visualize(tables[batch_idx], (row_start_bboxes[batch_idx], row_center_points[batch_idx], row_segm_logits[batch_idx], \
                    col_start_bboxes[batch_idx], col_center_points[batch_idx], col_segm_logits[batch_idx], \
                        mg_logits[batch_idx], num_rows[batch_idx], num_cols[batch_idx]), prefix)


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
    
    eval_checkpoint = os.path.join(cfg.work_dir, 'best_f1_model.pth')
    load_checkpoint(eval_checkpoint, model)
    logger.info('Load checkpoint from: %s' % eval_checkpoint)

    with torch.no_grad():
        valid(cfg, valid_dataloader, model)


if __name__ == '__main__':
    main()
