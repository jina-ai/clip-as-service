#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# solving chinese law-article classification problem: https://github.com/thunlp/CAIL/blob/master/README_en.md

import json
import os
import random

import GPUtil
import tensorflow as tf
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

from gpu_env import MODEL_ID
from service.client import BertClient

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
tf.logging.set_verbosity(tf.logging.INFO)

train_fp = ['/data/cips/data/lab/data/dataset/final_all_data/exercise_contest/data_train.json']
eval_fp = ['/data/cips/data/lab/data/dataset/final_all_data/exercise_contest/data_test.json']

batch_size = 128
num_parallel_calls = 4
num_concurrent_clients = num_parallel_calls * 2  # should be at least greater than `num_parallel_calls`

bc_clients = [BertClient(show_server_config=False, port=5557, port_out=5558) for _ in range(num_concurrent_clients)]

# hardcoded law_ids
laws = [184, 336, 314, 351, 224, 132, 158, 128, 223, 308, 341, 349, 382, 238, 369, 248, 266, 313, 127, 340, 288, 172,
        209, 243, 302, 200, 227, 155, 147, 143, 261, 124, 359, 343, 291, 241, 235, 367, 393, 274, 240, 269, 199, 119,
        246, 282, 133, 177, 170, 310, 364, 201, 312, 244, 357, 233, 236, 264, 225, 234, 328, 417, 151, 135, 136, 348,
        217, 168, 134, 237, 262, 150, 114, 196, 303, 191, 392, 226, 267, 272, 212, 353, 315, 205, 372, 215, 350, 275,
        385, 164, 338, 292, 159, 162, 333, 388, 356, 375, 326, 402, 397, 125, 395, 290, 176, 354, 185, 141, 279, 399,
        192, 383, 307, 295, 361, 286, 404, 390, 294, 115, 344, 268, 171, 117, 273, 193, 418, 220, 198, 231, 386, 363,
        346, 210, 270, 144, 347, 280, 281, 118, 122, 116, 360, 239, 228, 305, 130, 152, 389, 276, 213, 186, 413, 285,
        316, 245, 232, 175, 149, 263, 387, 283, 391, 211, 396, 352, 345, 258, 253, 163, 140, 293, 194, 342, 161, 358,
        271, 156, 260, 384, 153, 277, 214]

laws_str = [str(x) for x in laws]


def get_encodes(x):
    # x is `batch_size` of lines, each of which is a json object
    samples = [json.loads(l) for l in x]
    text = [s['fact'][:50] + s['fact'][-50:] for s in samples]
    # get a client from available clients
    bc_client = bc_clients.pop()
    features = bc_client.encode(text)
    # after use, put it back
    bc_clients.append(bc_client)
    # randomly choose a label
    labels = [[str(random.choice(s['meta']['relevant_articles']))] for s in samples]
    return features, labels


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
run_config = RunConfig(model_dir='/data/cips/save/%s' % MODEL_ID,
                       session_config=config,
                       save_checkpoints_steps=1000)

estimator = DNNClassifier(
    hidden_units=[512],
    feature_columns=[tf.feature_column.numeric_column('feature', shape=(768,))],
    n_classes=len(laws),
    config=run_config,
    label_vocabulary=laws_str,
    dropout=0.1)

input_fn = lambda fp: (tf.data.TextLineDataset(fp)
                       .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
                       .batch(batch_size)
                       .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string], name='bert_client'),
                            num_parallel_calls=num_parallel_calls)
                       .map(lambda x, y: ({'feature': x}, y))
                       .prefetch(20))

train_spec = TrainSpec(input_fn=lambda: input_fn(train_fp))
eval_spec = EvalSpec(input_fn=lambda: input_fn(eval_fp), throttle_secs=0)
train_and_evaluate(estimator, train_spec, eval_spec)
