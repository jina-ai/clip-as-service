# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from enum import Enum

import tensorflow as tf
from tensorflow.python.estimator.model_fn import EstimatorSpec

from . import modeling, tokenization


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


def minus_mask(x, mask, offset=1e30):
    """
    masking by subtract a very large number
    :param x: sequence data in the shape of [B, L, D]
    :param mask: 0-1 mask in the shape of [B, L]
    :param offset: very large negative number
    :return: masked x
    """
    return x - tf.expand_dims(1.0 - mask, axis=-1) * offset


def mul_mask(x, mask):
    """
    masking by multiply zero
    :param x: sequence data in the shape of [B, L, D]
    :param mask: 0-1 mask in the shape of [B, L]
    :return: masked x
    """
    return x * tf.expand_dims(mask, axis=-1)


def masked_reduce_max(x, mask):
    return tf.reduce_max(minus_mask(x, mask), axis=1)


def masked_reduce_mean(x, mask, jitter=1e-10):
    return tf.reduce_sum(mul_mask(x, mask), axis=1) / (tf.reduce_sum(mask, axis=1, keepdims=True) + jitter)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_type_ids):
        # self.unique_id = unique_id
        # self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def model_fn_builder(bert_config, init_checkpoint, use_one_hot_embeddings=False,
                     pooling_strategy=PoolingStrategy.REDUCE_MEAN,
                     pooling_layer=[-2]):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        client_id = features["client_id"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        all_layers = []
        if len(pooling_layer) == 1:
            encoder_layer = model.all_encoder_layers[pooling_layer[0]]
        else:
            for layer in pooling_layer:
                all_layers.append(model.all_encoder_layers[layer])
            encoder_layer = tf.concat(all_layers, -1)

        input_mask = tf.cast(input_mask, tf.float32)
        if pooling_strategy == PoolingStrategy.REDUCE_MEAN:
            pooled = masked_reduce_mean(encoder_layer, input_mask)
        elif pooling_strategy == PoolingStrategy.REDUCE_MAX:
            pooled = masked_reduce_max(encoder_layer, input_mask)
        elif pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
            pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                masked_reduce_max(encoder_layer, input_mask)], axis=1)
        elif pooling_strategy == PoolingStrategy.FIRST_TOKEN or pooling_strategy == PoolingStrategy.CLS_TOKEN:
            pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
        elif pooling_strategy == PoolingStrategy.LAST_TOKEN or pooling_strategy == PoolingStrategy.SEP_TOKEN:
            seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
            rng = tf.range(0, tf.shape(seq_len)[0])
            indexes = tf.stack([rng, seq_len - 1], 1)
            pooled = tf.gather_nd(encoder_layer, indexes)
        elif pooling_strategy == PoolingStrategy.NONE:
            pooled = encoder_layer
        else:
            raise NotImplementedError()

        predictions = {
            'client_id': client_id,
            'encodes': pooled
        }

        return EstimatorSpec(mode=mode, predictions=predictions)

    return model_fn


def convert_lst_to_features(lst_str, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    for (ex_index, example) in enumerate(read_examples(lst_str)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        # if ex_index < 5:
        #     tf.logging.info("*** Example ***")
        #     tf.logging.info("unique_id: %s" % (example.unique_id))
        #     tf.logging.info("tokens: %s" % " ".join(
        #         [tokenization.printable_text(x) for x in tokens]))
        #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     tf.logging.info(
        #         "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        yield InputFeatures(
            # unique_id=example.unique_id,
            # tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(lst_strs):
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for ss in lst_strs:
        line = tokenization.convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1
