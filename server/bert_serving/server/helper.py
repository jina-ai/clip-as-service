import argparse
import logging
import os
import uuid

import zmq
from zmq.utils import jsonapi

from .graph import PoolingStrategy

__all__ = ['set_logger', 'send_ndarray', 'get_args_parser', 'check_tf_version', 'auto_bind']


def set_logger(context):
    logger = logging.getLogger(context)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def send_ndarray(src, dest, X, req_id=b'', flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype=str(X.dtype), shape=X.shape)
    return src.send_multipart([dest, jsonapi.dumps(md), X, req_id], flags, copy=copy, track=track)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', type=str, required=True,
                        help='directory of a pretrained BERT model')
    parser.add_argument('-max_seq_len', type=int, default=25,
                        help='maximum length of a sequence')
    parser.add_argument('-num_worker', type=int, default=1,
                        help='number of server instances')
    parser.add_argument('-max_batch_size', type=int, default=256,
                        help='maximum number of sequences handled by each worker')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    parser.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for outputting result to client')
    parser.add_argument('-pooling_layer', type=int, nargs='+', default=[-2],
                        help='the encoder layer(s) that receives pooling. '
                             'Give a list in order to concatenate several layers into 1.')
    parser.add_argument('-pooling_strategy', type=PoolingStrategy.from_string,
                        default=PoolingStrategy.REDUCE_MEAN, choices=list(PoolingStrategy),
                        help='the pooling strategy for generating encoding vectors')
    parser.add_argument('-cpu', action='store_true', default=False,
                        help='running on CPU (default is on GPU)')
    parser.add_argument('-xla', action='store_true', default=False,
                        help='enable XLA compiler')
    parser.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determines the fraction of the overall amount of memory '
                             'that each visible GPU should be allocated per worker. '
                             'Should be in range [0.0, 1.0]')
    parser.add_argument('-version', action='store_true', default=False,
                        help='show version and exit')
    return parser


def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    assert int(tf_ver[0]) >= 1 and int(tf_ver[1]) >= 10, 'Tensorflow >=1.10 is required!'
    return tf_ver


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
