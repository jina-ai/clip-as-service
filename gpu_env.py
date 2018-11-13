#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import os
from datetime import datetime
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

IGNORE_PATTERNS = ('data', '*.pyc', 'CVS', '.git', 'tmp', '.svn', '__pycache__', '.gitignore', '.*.yaml')
MODEL_ID = datetime.now().strftime("%m%d-%H%M%S") + (
    os.environ['suffix_model_id'] if 'suffix_model_id' in os.environ else '')
APP_NAME = 'bert'


class SummaryType(Enum):
    SCALAR = 1
    HISTOGRAM = 2
    SAMPLED = 3


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    INFER = 3
    INTERACT = 4
    INIT_LAW_EMBED = 5
    BOTTLENECK = 6
    COMPETITION = 7
    ENSEMBLE = 8
