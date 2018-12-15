import argparse
import contextlib
import json
import logging
import os
import tempfile
import uuid

import zmq
from zmq.utils import jsonapi

from .bert import modeling
from .bert.extract_features import masked_reduce_mean, PoolingStrategy, \
    masked_reduce_max, mul_mask

__all__ = ['set_logger', 'send_ndarray', 'optimize_graph',
           'get_args_parser', 'check_tf_version', 'auto_bind']


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


def optimize_graph(args):
    # we don't need GPU for optimizing the graph
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

    config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

    config_fp = os.path.join(args.model_dir, 'bert_config.json')
    init_checkpoint = os.path.join(args.model_dir, 'bert_model.ckpt')
    with tf.gfile.GFile(config_fp, 'r') as f:
        bert_config = modeling.BertConfig.from_dict(json.load(f))

    # input placeholders, not sure if they are friendly to XLA
    input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
    input_type_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_type_ids')

    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope if args.xla else contextlib.suppress

    with jit_scope():
        input_tensors = [input_ids, input_mask, input_type_ids]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=False)

        tvars = tf.trainable_variables()

        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        with tf.variable_scope("pooling"):
            if len(args.pooling_layer) == 1:
                encoder_layer = model.all_encoder_layers[args.pooling_layer[0]]
            else:
                all_layers = [model.all_encoder_layers[l] for l in args.pooling_layer]
                encoder_layer = tf.concat(all_layers, -1)

            input_mask = tf.cast(input_mask, tf.float32)
            if args.pooling_strategy == PoolingStrategy.REDUCE_MEAN:
                pooled = masked_reduce_mean(encoder_layer, input_mask)
            elif args.pooling_strategy == PoolingStrategy.REDUCE_MAX:
                pooled = masked_reduce_max(encoder_layer, input_mask)
            elif args.pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
                pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                    masked_reduce_max(encoder_layer, input_mask)], axis=1)
            elif args.pooling_strategy == PoolingStrategy.FIRST_TOKEN or \
                    args.pooling_strategy == PoolingStrategy.CLS_TOKEN:
                pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
            elif args.pooling_strategy == PoolingStrategy.LAST_TOKEN or \
                    args.pooling_strategy == PoolingStrategy.SEP_TOKEN:
                seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                rng = tf.range(0, tf.shape(seq_len)[0])
                indexes = tf.stack([rng, seq_len - 1], 1)
                pooled = tf.gather_nd(encoder_layer, indexes)
            elif args.pooling_strategy == PoolingStrategy.NONE:
                pooled = mul_mask(encoder_layer, input_mask)
            else:
                raise NotImplementedError()

        pooled = tf.identity(pooled, 'final_encodes')

        output_tensors = [pooled]
        tmp_g = tf.get_default_graph().as_graph_def()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_g = tf.graph_util.convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])
        dtypes = [n.dtype for n in input_tensors]
        tmp_g = optimize_for_inference(
            tmp_g,
            [n.name[:-2] for n in input_tensors],
            [n.name[:-2] for n in output_tensors],
            [dtype.as_datatype_enum for dtype in dtypes],
            False)

    tmp_file = tempfile.NamedTemporaryFile('w', delete=False).name
    with tf.gfile.GFile(tmp_file, 'wb') as f:
        f.write(tmp_g.SerializeToString())
    return tmp_file


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
