#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# using BertClient inside tf.data API

import json
import os
import time

import GPUtil
import tensorflow as tf
from bert_serving.client import BertClient

from plugin.quantizer.base_quantizer import BaseQuantizer

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable())

train_fp = ['/data/cips/data/larry-autoencoder/cail_0518/data_train.json']
batch_size = 256
num_parallel_calls = 4
num_concurrent_clients = 10  # should be greater than `num_parallel_calls`

bc_clients = [BertClient(port=5500, port_out=5501) for _ in range(num_concurrent_clients)]


def get_encodes(x):
    # x is `batch_size` of lines, each of which is a json object
    samples = [json.loads(l) for l in x]
    text = [s['fact'][-50:] for s in samples]
    # get a client from available clients
    bc_client = bc_clients.pop()
    features = bc_client.encode(text)
    # after use, put it back
    bc_clients.append(bc_client)
    labels = [0 for _ in text]
    return features, labels


data_node = (tf.data.TextLineDataset(train_fp).batch(batch_size)
             .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.int64], name='bert_client'),
                  num_parallel_calls=num_parallel_calls)
             .map(lambda x, y: {'feature': x, 'label': y})
             .make_one_shot_iterator().get_next())

quantizer = BaseQuantizer()
train_op = quantizer.train_op

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cnt, num_samples, start_t = 0, 0, time.perf_counter()
    while True:
        x = sess.run(data_node)
        _, loss = sess.run([quantizer.loss, quantizer.train_op], feed_dict={quantizer.ph_x: x['feature']})
        cnt += 1
        print('%10d: %.5f' % (cnt, loss))
