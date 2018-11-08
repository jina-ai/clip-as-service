import os
from datetime import datetime
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

IGNORE_PATTERNS = ('data', '*.pyc', 'CVS', '.git', 'tmp', '.svn', '__pycache__', '.gitignore', '.*.yaml')
MODEL_ID = datetime.now().strftime("%m%d-%H%M%S") + (
    os.environ['suffix_model_id'] if 'suffix_model_id' in os.environ else '')
APP_NAME = 'mrc'
NUM_PROC = 16


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


try:
    import GPUtil

    # GPUtil.showUtilization()
    DEVICE_ID_LIST = GPUtil.getFirstAvailable(order='random', maxMemory=0.5, maxLoad=0.1)
    DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    CONFIG_SET = 'default_gpu'
except FileNotFoundError:
    print('no gpu found!')
    DEVICE_ID = 'x'
    CONFIG_SET = 'default'
except RuntimeError:
    print('all gpus are occupied!')
    DEVICE_ID = '?'
    CONFIG_SET = 'default_gpu'

print('use config: %s' % CONFIG_SET)
