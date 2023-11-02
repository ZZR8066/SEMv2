import cv2
import sys
import json

from torch.nn.functional import normalize
sys.path.append('./')
sys.path.append('../')
import os
import tqdm
import torch
import numpy as np
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data.transform import CallImageNormalize
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1
from libs.utils.visualize import visualize
from libs.utils.checkpoint import load_checkpoint
from libs.utils.comm import synchronize, all_gather


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrc", type=str, default=None)
    parser.add_argument("--cfg", type=str, default='debug')
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    setup_config(args.cfg)
    if args.lrc is not None:
        cfg.valid_lrc_path = args.lrc
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    if args.image_dir is not None:
        cfg.image_dir = args.image_dir

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


def valid(cfg, image_files, model):
    model.eval()
    
    vis_dir = os.path.join(cfg.work_dir, 'infer_folder')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    vis_dir = os.path.abspath(vis_dir)
    
    image_normalize = CallImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for it, image_file in enumerate(tqdm.tqdm(image_files)):

        # transform to input type
        table = dict()
        image = cv2.imread(image_file)
        table.update(img=image)
        image_size = image.shape[:2][::-1] # w, h
        image_size = torch.tensor(image_size).long()
        image, *_ = image_normalize(image, None, None, None)
        images = image.unsqueeze(0).to(cfg.device)
        images_size = image_size.unsqueeze(0).to(cfg.device)
        tables = [table]
        
        # pred
        _, row_start_bboxes, row_segm_logits, \
            col_start_bboxes, col_segm_logits, \
                mg_logits, num_rows, num_cols = model(images, images_size)

        for batch_idx in range(len(tables)):
            prefix =os.path.join(os.path.abspath(vis_dir), str(int(it * len(tables) + batch_idx)))
            visualize(tables[batch_idx], (row_start_bboxes[batch_idx], row_segm_logits[batch_idx], \
                    col_start_bboxes[batch_idx], col_segm_logits[batch_idx], \
                        mg_logits[batch_idx], num_rows[batch_idx], num_cols[batch_idx]), prefix)


def main():
    init()

    import glob
    image_files = glob.glob(os.path.join(cfg.image_dir, '*.png'))
    logger.info('Inference image files have %d samples' % len(image_files))

    model = build_model(cfg)
    model.cuda()
    
    eval_checkpoint = os.path.join(cfg.work_dir, 'best_f1_model.pth')
    load_checkpoint(eval_checkpoint, model)
    logger.info('Load checkpoint from: %s' % eval_checkpoint)

    with torch.no_grad():
        valid(cfg, image_files, model)


if __name__ == '__main__':
    main()
