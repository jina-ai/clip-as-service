import argparse
import logging
import os
import sys
import uuid
from http.server import SimpleHTTPRequestHandler

import zmq
from zmq.utils import jsonapi

__all__ = ['set_logger', 'send_ndarray', 'get_args_parser',
           'check_tf_version', 'auto_bind', 'import_tf', 'BertRequestHandler']


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
                        help='filename of the checkpoint file. By default it is "bert_model.ckpt", but \
                             for a fine-tuned model the name could be different.')
    group1.add_argument('-config_name', type=str, default='bert_config.json',
                        help='filename of the JSON config file for BERT model.')
    group1.add_argument('-graph_tmp_dir', type=str, default=None,
                        help='path to graph temp file')

    group2 = parser.add_argument_group('BERT Parameters',
                                       'config how BERT model and pooling works')
    group2.add_argument('-max_seq_len', type=int, default=25,
                        help='maximum length of a sequence')
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

    group3 = parser.add_argument_group('Serving Configs',
                                       'config how server utilizes GPU/CPU resources')
    group3.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    group3.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for sending result to client')
    group3.add_argument('-http_port', type=int, default=None,
                        help='server port for receiving HTTP requests')
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


class BertRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _set_headers(self, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', self.server.args.cors)
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        if self.path == '/status':
            self.server.logger.info('checking server status')
            self._response_dict(self.server.bc.server_status, code=200)
        if self.path == '/terminate':
            self.server.logger.info('shutting down HTTP server')
            self.server.shutdown()
            self.server.logger.info('you can no longer make HTTP request to this server')
            self._response_msg('you can no longer make HTTP request to this server', code=410)

    def do_POST(self):
        try:
            self.server.logger.info('new request [%s] %s' % (self.log_date_time_string(), self.client_address))
            content_len = int(self.headers.get('Content-Length', 0))
            content_type = self.headers.get('Content-Type', 'application/json')
            if content_len and content_type == 'application/json':
                post_body = self.rfile.read(content_len)
                data = jsonapi.loads(post_body)
                result = self.server.bc.encode(data['texts'])
                self._response_dict({'id': data['id'], 'result': result}, code=200)
            else:
                raise TypeError('"Content-Length" or "Content-Type" are wrong')
        except Exception as e:
            self._response_msg(str(e), msg_type=e.__class__.__name__, code=400)
            self.server.logger.error('error when handling HTTP request', exc_info=True)

    def log_message(self, format, *args):
        self.server.logger.info('%s [%s] %s' % (self.address_string(),
                                                self.log_date_time_string(),
                                                format % args))

    def _response_dict(self, x, code=200):
        self._set_headers(code)
        self.wfile.write(jsonapi.dumps(x, ensure_ascii=False))
        self.wfile.flush()
        self.server.logger.info('send result back')

    def _response_msg(self, msg, msg_type=RuntimeError.__class__.__name__, code=200):
        self._response_dict({'type': msg_type, 'message': msg})
