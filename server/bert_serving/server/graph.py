import contextlib
import json
import os
import tempfile
from enum import Enum

from termcolor import colored

from .bert import modeling
from .helper import import_tf, set_logger

__all__ = ['PoolingStrategy', 'optimize_graph']


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


def optimize_graph(args, logger=None):
    if not logger:
        logger = set_logger(colored('GRAPHOPT', 'cyan'), args.verbose)
    try:
        # we don't need GPU for optimizing the graph
        tf = import_tf(verbose=args.verbose)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

        config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

        config_fp = os.path.join(args.model_dir, args.config_name)
        init_checkpoint = os.path.join(args.tuned_model_dir or args.model_dir, args.ckpt_name)
        logger.info('model config: %s' % config_fp)
        logger.info(
            'checkpoint%s: %s' % (' (override by fine-tuned model)' if args.tuned_model_dir else '', init_checkpoint))
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))

        logger.info('build graph...')
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

            minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

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
            logger.info('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            logger.info('freeze...')
            tmp_g = tf.graph_util.convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])
            dtypes = [n.dtype for n in input_tensors]
            logger.info('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)
        tmp_file = tempfile.NamedTemporaryFile('w', delete=False).name
        logger.info('write graph to a tmp file: %s' % tmp_file)
        with tf.gfile.GFile(tmp_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return tmp_file
    except Exception as e:
        logger.error('fail to optimize the graph!')
        logger.error(e)
