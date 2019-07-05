import argparse
import logging
import os
import sys
import time
import uuid
import warnings
from collections import OrderedDict


import zmq
from termcolor import colored
from zmq.utils import jsonapi

__all__ = ['set_logger', 'send_ndarray', 'get_args_parser',
           'check_tf_version', 'auto_bind', 'import_tf', 'TimeContext', 'CappedHistogram']


def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


def send_ndarray(src, dest, X, req_id=b'', flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype=str(X.dtype), shape=X.shape)
    return src.send_multipart([dest, jsonapi.dumps(md), X, req_id], flags, copy=copy, track=track)


def check_max_seq_len(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        ivalue = int(value)
        if ivalue <= 3:
            raise argparse.ArgumentTypeError("%s is an invalid int value must be >3 "
                                             "(account for maximum three special symbols in BERT model) or NONE" % value)
    except TypeError:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue


def get_args_parser():
    from . import __version__
    from .graph import PoolingStrategy

    parser = argparse.ArgumentParser(description='Start a BertServer for serving')

    group1 = parser.add_argument_group('File Paths',
                                       'config the path, checkpoint and filename of a pretrained/fine-tuned BERT model')
    group1.add_argument('-model_dir', type=str, required=True,
                        help='directory of a pretrained BERT model')
    group1.add_argument('-tuned_model_dir', type=str,
                        help='directory of a fine-tuned BERT model')
    group1.add_argument('-ckpt_name', type=str, default='bert_model.ckpt',
                        help='filename of the checkpoint file. By default it is "bert_model.ckpt", but \
                             for a fine-tuned model the name could be different.')
    group1.add_argument('-config_name', type=str, default='bert_config.json',
                        help='filename of the JSON config file for BERT model.')
    group1.add_argument('-graph_tmp_dir', type=str, default=None,
                        help='path to graph temp file')

    group2 = parser.add_argument_group('BERT Parameters',
                                       'config how BERT model and pooling works')
    group2.add_argument('-max_seq_len', type=check_max_seq_len, default=25,
                        help='maximum length of a sequence, longer sequence will be trimmed on the right side. '
                             'set it to NONE for dynamically using the longest sequence in a (mini)batch.')
    group2.add_argument('-cased_tokenization', dest='do_lower_case', action='store_false', default=True,
                        help='Whether tokenizer should skip the default lowercasing and accent removal.'
                             'Should be used for e.g. the multilingual cased pretrained BERT model.')
    group2.add_argument('-pooling_layer', type=int, nargs='+', default=[-2],
                        help='the encoder layer(s) that receives pooling. \
                        Give a list in order to concatenate several layers into one')
    group2.add_argument('-pooling_strategy', type=PoolingStrategy.from_string,
                        default=PoolingStrategy.REDUCE_MEAN, choices=list(PoolingStrategy),
                        help='the pooling strategy for generating encoding vectors')
    group2.add_argument('-mask_cls_sep', action='store_true', default=False,
                        help='masking the embedding on [CLS] and [SEP] with zero. \
                        When pooling_strategy is in {CLS_TOKEN, FIRST_TOKEN, SEP_TOKEN, LAST_TOKEN} \
                        then the embedding is preserved, otherwise the embedding is masked to zero before pooling')
    group2.add_argument('-show_tokens_to_client', action='store_true', default=False,
                        help='sending tokenization results to client')

    group3 = parser.add_argument_group('Serving Configs',
                                       'config how server utilizes GPU/CPU resources')
    group3.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    group3.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for sending result to client')
    group3.add_argument('-http_port', type=int, default=None,
                        help='server port for receiving HTTP requests')
    group3.add_argument('-http_max_connect', type=int, default=10,
                        help='maximum number of concurrent HTTP connections')
    group3.add_argument('-cors', type=str, default='*',
                        help='setting "Access-Control-Allow-Origin" for HTTP requests')
    group3.add_argument('-num_worker', type=int, default=1,
                        help='number of server instances')
    group3.add_argument('-max_batch_size', type=int, default=256,
                        help='maximum number of sequences handled by each worker')
    group3.add_argument('-priority_batch_size', type=int, default=16,
                        help='batch smaller than this size will be labeled as high priority,'
                             'and jumps forward in the job queue')
    group3.add_argument('-cpu', action='store_true', default=False,
                        help='running on CPU (default on GPU)')
    group3.add_argument('-xla', action='store_true', default=False,
                        help='enable XLA compiler (experimental)')
    group3.add_argument('-fp16', action='store_true', default=False,
                        help='use float16 precision (experimental)')
    group3.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determine the fraction of the overall amount of memory \
                        that each visible GPU should be allocated per worker. \
                        Should be in range [0.0, 1.0]')
    group3.add_argument('-device_map', type=int, nargs='+', default=[],
                        help='specify the list of GPU device ids that will be used (id starts from 0). \
                        If num_worker > len(device_map), then device will be reused; \
                        if num_worker < len(device_map), then device_map[:num_worker] will be used')
    group3.add_argument('-prefetch_size', type=int, default=10,
                        help='the number of batches to prefetch on each worker. When running on a CPU-only machine, \
                        this is set to 0 for comparability')
    group3.add_argument('-fixed_embed_length', action='store_true', default=False,
                        help='when "max_seq_len" is set to None, the server determines the "max_seq_len" according to '
                             'the actual sequence lengths within each batch. When "pooling_strategy=NONE", '
                             'this may cause two ".encode()" from the same client results in different sizes [B, T, D].'
                             'Turn this on to fix the "T" in [B, T, D] to "max_position_embeddings" in bert json config.')

    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    return parser


def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    if int(tf_ver[0]) <= 1 and int(tf_ver[1]) < 10:
        raise ModuleNotFoundError('Tensorflow >=1.10 (one-point-ten) is required!')
    elif int(tf_ver[0]) > 1:
        warnings.warn('Tensorflow %s is not tested! It may or may not work. '
                      'Feel free to submit an issue at https://github.com/hanxiao/bert-as-service/issues/' % tf.__version__)
    return tf_ver


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def auto_bind(socket):
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://127.0.0.1')
    else:
        # Get the location for tmp file for sockets
        try:
            tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
            if not os.path.exists(tmp_dir):
                raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
            tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
        except KeyError:
            tmp_dir = '*'

        socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')


def get_run_args(parser_fn=get_args_parser, printed=True):
    args = parser_fn().parse_args()
    if printed:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args


def get_benchmark_parser():
    parser = get_args_parser()
    parser.description = 'Benchmark BertServer locally'

    parser.set_defaults(num_client=1, client_batch_size=4096)

    group = parser.add_argument_group('Benchmark parameters', 'config the experiments of the benchmark')

    group.add_argument('-test_client_batch_size', type=int, nargs='*', default=[1, 16, 256, 4096])
    group.add_argument('-test_max_batch_size', type=int, nargs='*', default=[8, 32, 128, 512])
    group.add_argument('-test_max_seq_len', type=int, nargs='*', default=[32, 64, 128, 256])
    group.add_argument('-test_num_client', type=int, nargs='*', default=[1, 4, 16, 64])
    group.add_argument('-test_pooling_layer', type=int, nargs='*', default=[[-j] for j in range(1, 13)])

    group.add_argument('-wait_till_ready', type=int, default=30,
                       help='seconds to wait until server is ready to serve')
    group.add_argument('-client_vocab_file', type=str, default='README.md',
                       help='file path for building client vocabulary')
    group.add_argument('-num_repeat', type=int, default=10,
                       help='number of repeats per experiment (must >2), '
                            'as the first two results are omitted for warm-up effect')
    return parser


def get_shutdown_parser():
    parser = argparse.ArgumentParser()
    parser.description = 'Shutting down a BertServer instance running on a specific port'

    parser.add_argument('-ip', type=str, default='localhost',
                        help='the ip address that a BertServer is running on')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, required=True,
                        help='the port that a BertServer is running on')
    parser.add_argument('-timeout', type=int, default=5000,
                        help='timeout (ms) for connecting to a server')
    return parser


class TimeContext:
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        self.start = time.perf_counter()
        print(self._msg, end=' ...\t', flush=True)

    def __exit__(self, typ, value, traceback):
        self.duration = time.perf_counter() - self.start
        print(colored('    [%3.3f secs]' % self.duration, 'green'), flush=True)

class CappedHistogram:
    """Space capped dict with aggregate stat tracking.

    Evicts using LRU policy when at capacity; evicted elements are added to aggregate stats.
    Arguments:
    capacity -- the capacity limit of the dict
    """
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.base_bins = 0
        self.base_count = 0
        self.base_min = float('inf')
        self.min_count = 0
        self.base_max = 0
        self.max_count = 0

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        return 0

    def __setitem__(self, key, value):
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self._evict()

    def total_size(self):
        return self.base_bins + len(self.cache)

    def __len__(self):
        return len(self.cache)

    def values(self):
        return self.cache.values()

    def _evict(self):
        key,val = self.cache.popitem(False)
        self.base_bins += 1
        self.base_count += val
        if val < self.base_min:
            self.base_min = val
            self.min_count = 1
        elif val == self.base_min:
            self.min_count += 1
        if val > self.base_max:
            self.base_max = val
            self.max_count = 1
        elif val == self.base_max:
            self.max_count += 1

    def get_stat_map(self, name):
        if len(self.cache) == 0:
            return {}
        counts = self.cache.values()
        avg = (self.base_count + sum(counts)) / (self.base_bins + len(counts))
        min_, max_ = min(counts), max(counts)
        num_min, num_max = 0, 0
        if self.base_min <= min_:
            min_ = self.base_min
            num_min += self.min_count
        if self.base_min >= min_:
            num_min += sum(v == min_ for v in counts)

        if self.base_max >= max_:
            max_ = self.base_max
            num_max += self.max_count
        if self.base_max <= max_:
            num_max += sum(v == max_ for v in counts)

        return {
            'avg_%s' % name: avg,
            'min_%s' % name: min_,
            'max_%s' % name: max_,
            'num_min_%s' % name: num_min,
            'num_max_%s' % name: num_max,
        }