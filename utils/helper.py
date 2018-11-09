import importlib
import logging
import math
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from random import shuffle

import GPUtil
import tensorflow as tf
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from tensorflow.contrib.training import HParams
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from gpu_env import APP_NAME, DEVICE_ID, IGNORE_PATTERNS

millnames = ['', ' K', ' M', ' BL', ' TL']
regex_title_source = re.compile(r'^([^_\-—]*).*?[_\-—]\s?([^_\-—]+)[\s_\-—]?$')


def set_logger(model_id=None):
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)
    if model_id:
        formatter = logging.Formatter(
            '%(levelname)-.1s:' + model_id + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
            '%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(
            '%(levelname)-.1s:[%(filename)s:%(lineno)d]:%(message)s', datefmt=
            '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def touch(fname: str, times=None, create_dirs: bool = False):
    import os
    if create_dirs:
        base_dir = os.path.dirname(fname)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    with open(fname, 'a'):
        os.utime(fname, times)


def touch_dir(base_dir: str) -> None:
    import os
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def millify(n):
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


def args2hparam(args, vocab):
    params = vars(args)
    params['vocab'] = vocab
    p = HParams()
    for k, v in params.items():
        p.add_hparam(k, v)
    return p


def runner(main, *done):
    logger = logging.getLogger(APP_NAME)
    try:
        main()
    except (tf.errors.OutOfRangeError, IndexError) as e:
        logger.warning('Data has been exhausted! Done!')
    finally:
        [f() for f in done]


def parse_args(yaml_path, model_id, default_set, followup=None, use_as_component=False):
    logger = logging.getLogger(APP_NAME)

    hparams = HParams()
    hparams.add_hparam('model_id', model_id)

    with open('default.yaml') as fp:
        configs = YAML().load(fp)
        default_cfg = configs[default_set]

        add_param_recur(hparams, default_cfg)

        if yaml_path:
            logger.info('loading parameters...')
            with open(yaml_path) as fp:
                customized = YAML().load(fp)
                for k, v in customized.items():
                    if k in hparams and hparams.get(k) != v:
                        logger.info('%20s: %20s -> %20s' % (k, hparams.get(k), v))
                        hparams.set_hparam(k, v)
                    elif k not in hparams:
                        logger.warning('%s is not a valid attribute! ignore!' % k)

    if followup:
        # useful when changing args for prediction
        logger.info('override args with follow-up args...')
        for k, v in followup.items():
            if k in hparams and hparams.get(k) != v:
                logger.info('%20s: %20s -> %20s' % (k, hparams.get(k), v))
                hparams.set_hparam(k, v)
            elif k not in hparams:
                logger.warning('%s is not a valid attribute! ignore!' % k)

    hparams.add_hparam('save_dir', os.path.join(hparams.get('model_dir'), hparams.get('model_id')))
    hparams.set_hparam('summary_dir', os.path.join(hparams.get('save_dir'), 'summary'))
    hparams.add_hparam('code_dir', os.path.join(hparams.get('save_dir'), 'code'))

    if not (hparams.get('use_as_component') or use_as_component):
        # reset logger model id
        logger = set_logger(model_id='%s:%s' % (DEVICE_ID, hparams.get('model_id')))

        if hparams.get('backup_code'):
            try:
                logger.info('copying current code base to %s ...' % hparams.get('save_dir'))
                shutil.copytree('./', hparams.get('code_dir'), ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
            except FileExistsError:
                logger.info('code base exist, no need to copy!')
    else:
        hparams.set_hparam('use_as_component', True)

    # if hparams.get('model_id') != model_id:
    #     logger.warning('model id is changed %s -> %s! '
    #                    'This happens when you train a pretrained model' % (
    #                        hparams.get('model_id'), model_id))
    #     hparams.set_hparam('model_id', model_id)

    hparams.add_hparam('loss_csv_file', os.path.join(hparams.get('save_dir'), 'loss.csv'))
    hparams.add_hparam('is_serving', False)

    logger.info('current parameters')
    for k, v in sorted(vars(hparams).items()):
        if not k.startswith('_'):
            logger.info('%20s = %-20s' % (k, v))

    return hparams


def add_param_recur(root, p_tree):
    for k, v in p_tree.items():
        if isinstance(v, CommentedMap):
            new_node = HParams()
            add_param_recur(new_node, v)
            root.add_hparam(k, new_node)
        else:
            root.add_hparam(k, v)


def fill_gpu_jobs(all_jobs, logger, job_parser,
                  wait_until_next=300, retry_delay=300, do_shuffle=False):
    if do_shuffle:
        shuffle(all_jobs)
    all_procs = []

    while all_jobs:
        logger.info('number of jobs in the queue: %d' % len(all_jobs))
        j = all_jobs.pop()
        logger.info('will start the job: %s ...' % job_parser(j))

        try:
            GPUtil.getFirstAvailable()
            # check if there is a free GPU!
            process = subprocess.Popen(job_parser(j), shell=True)
            all_procs.append((process, j))
            time.sleep(wait_until_next)
        except FileNotFoundError:
            logger.warning('there is no gpu, running on cpu!')
            process = subprocess.Popen(job_parser(j), shell=True)
            all_procs.append((process, j))
        except RuntimeError as e:
            logger.error(str(e))
            logger.warning('all gpus are busy! waiting for a free slot...')
            # add job back
            all_jobs.append(j)
            time.sleep(retry_delay)

    exit_codes = [(p.wait(), j) for p, j in all_procs]
    return [v for p, v in exit_codes if p != 0]


def get_args_cli(args):
    d = defaultdict(list)
    if args:
        for k, v in ((k.lstrip('-'), v) for k, v in (a.split('=') for a in args)):
            d[k].append(v)
        for k, v in d.items():
            parsed_v = [s for s in (parse_arg(vv) for vv in v) if s is not None]
            if len(parsed_v) > 1:
                d[k] = parsed_v
            if len(parsed_v) == 1:
                d[k] = parsed_v[0]
    return d


def parse_arg(v: str):
    if v.startswith('[') and v.endswith(']'):
        # function args must be immutable tuples not list
        tmp = v.replace('[', '').replace(']', '').strip().split(',')
        if len(tmp) > 0:
            return [parse_arg(vv.strip()) for vv in tmp]
        else:
            return []
    try:
        v = int(v)  # parse int parameter
    except ValueError:
        try:
            v = float(v)  # parse float parameter
        except ValueError:
            if len(v) == 0:
                # ignore it when the parameter is empty
                v = None
            elif v.lower() == 'true':  # parse boolean parameter
                v = True
            elif v.lower() == 'false':
                v = False
    return v


def get_scope_name():
    return tf.get_variable_scope().name.split('/')[0]


def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
    """
    negative log likelihood loss
    """
    with tf.name_scope(scope, "log_loss"):
        labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1, dtype=tf.float32)
        losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
    return losses


def normalize_distribution(p, eps=1e-9):
    p += eps
    norm = tf.reduce_sum(p, axis=1)
    return tf.cast(p, tf.float32) / tf.reshape(norm, (-1, 1))


def kl_divergence(p, q, eps=1e-9):
    p = normalize_distribution(p, eps)
    q = normalize_distribution(q, eps)
    return tf.reduce_sum(p * tf.log(p / q), axis=1)


def get_kl_loss(start_label, start_probs, bandwidth=1.0):
    a = tf.reshape(tf.range(tf.shape(start_probs)[1]), (1, -1))
    b = tf.reshape(start_label, (-1, 1))
    start_true_probs = tf.exp(-tf.cast(tf.squared_difference(a, b), tf.float32) / bandwidth)
    return sym_kl_divergence(start_true_probs, start_probs)


def sym_kl_divergence(p, q, eps=1e-9):
    return (kl_divergence(p, q, eps) + kl_divergence(q, p, eps)) / 2.0


def get_conv1d(x, out_dim, window_len, name, act_fn):
    return tf.layers.conv1d(x, out_dim, window_len, strides=1, padding='SAME', name=name, activation=act_fn)


def upsampling_a2b(a, b, D_a):
    return tf.squeeze(tf.image.resize_images(tf.expand_dims(a, axis=-1), [tf.shape(b)[1], D_a],
                                             method=ResizeMethod.NEAREST_NEIGHBOR), axis=-1)


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def get_tmp_yaml(par, prefix=None):
    import tempfile
    with tempfile.NamedTemporaryFile('w', delete=False, prefix=prefix) as tmp:
        YAML().dump(par, tmp)
        return tmp.name


def build_model(args, reset_graph=True):
    rccore = importlib.import_module(args.package_rccore)
    if reset_graph:
        tf.reset_default_graph()
    return rccore.RCCore(args)


class JobContext:
    def __init__(self, msg, logger=None):
        self._msg = msg
        self._logger = logger

    def __enter__(self):
        self.start = time.clock()
        if not self._logger:
            print(self._msg, end='')
        else:
            self._logger.info('☐ %s' % self._msg)

    def __exit__(self, typ, value, traceback):
        self.duration = time.clock() - self.start
        if not self._logger:
            print('    [%.3f secs]\n' % self.duration)
        else:
            self._logger.info('☑ %s    [%.3f secs]' % (self._msg, self.duration))
