# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

from .comm import get_rank


_default_logger = None


def __init_logger():
    global _default_logger
    if get_rank() == 0:
        logger = logging.getLogger('default')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

        if not any([isinstance(item, logging.StreamHandler) for item in logger.handlers]):
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        _default_logger = logger


__init_logger()


def setup_logger(name, save_dir, filename="log.txt"):
    global _default_logger
    # don't log results for the non-master process
    if get_rank() == 0:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

        if not any([isinstance(item, logging.StreamHandler) for item in logger.handlers]):
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        logger.handlers = [item for item in logger.handlers if not isinstance(item, logging.FileHandler)]
        if save_dir:
            log_path = os.path.join(save_dir, filename)
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path))
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        _default_logger = logger


def info(*args, **kwargs):
    if get_rank() == 0:
        _default_logger.info(*args, **kwargs)


def error(*args, **kwargs):
    if get_rank() == 0:
        _default_logger.error(*args, **kwargs)
