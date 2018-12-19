import argparse
import logging
import os
import uuid

import zmq
from zmq.utils import jsonapi

__all__ = ['set_logger', 'send_ndarray', 'get_args_parser', 'check_tf_version', 'auto_bind', 'import_tf']


def set_logger(context, verbose=False):
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


def send_ndarray(src, dest, X, req_id=b'', flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype=str(X.dtype), shape=X.shape)
    return src.send_multipart([dest, jsonapi.dumps(md), X, req_id], flags, copy=copy, track=track)


def get_args_parser():
    from . import __version__
    from .graph import PoolingStrategy

    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('File Paths',
                                       'config the path, checkpoint and filename of a pretrained/fine-tuned BERT model')
    group1.add_argument('-model_dir', type=str, required=True,
                        help='directory of a pretrained BERT model')
    group1.add_argument('-tuned_model_dir', type=str,
                        help='directory of a fine-tuned BERT model')
    group1.add_argument('-ckpt_name', type=str, default='bert_model.ckpt',
                        help='filename of the checkpoint file. By default it is "bert_model.ckpt", but '
                             'for a fine-tuned model the name could be different.')
    group1.add_argument('-config_name', type=str, default='bert_config.json',
                        help='filename of the JSON config file for BERT model.')

    group2 = parser.add_argument_group('BERT Parameters',
                                       'config how BERT model and pooling works')
    group2.add_argument('-max_seq_len', type=int, default=25,
                        help='maximum length of a sequence')
    group2.add_argument('-pooling_layer', type=int, nargs='+', default=[-2],
                        help='the encoder layer(s) that receives pooling. '
                             'Give a list in order to concatenate several layers into one')
    group2.add_argument('-pooling_strategy', type=PoolingStrategy.from_string,
                        default=PoolingStrategy.REDUCE_MEAN, choices=list(PoolingStrategy),
                        help='the pooling strategy for generating encoding vectors')

    group3 = parser.add_argument_group('Serving Configs',
                                       'config how server utilizes GPU/CPU resources')
    group3.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    group3.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for sending result to client')

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
    group3.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determine the fraction of the overall amount of memory '
                             'that each visible GPU should be allocated per worker. '
                             'Should be in range [0.0, 1.0]')
    group3.add_argument('-device_map', type=int, nargs='+', default=[],
                        help='specify the list of GPU device ids that will be used (id starts from 0).'
                             'If num_worker > len(device_map), then device will be reused; '
                             'if num_worker < len(device_map), then device_map[:num_worker] will be used')

    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    return parser


def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    assert int(tf_ver[0]) >= 1 and int(tf_ver[1]) >= 10, 'Tensorflow >=1.10 is required!'
    return tf_ver


def import_tf(device_id=-1, verbose=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def auto_bind(socket):
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://*')
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
