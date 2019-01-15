#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

import json
import os
import random

import tensorflow as tf
from bert_serving.client import BertClient

from plugin.quantizer.base_quantizer import BaseQuantizer

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

train_fp = ['/data/cips/data/larry-autoencoder/cail_0518/data_train.json']
dev_fp = ['/data/cips/data/larry-autoencoder/cail_0518/data_valid.json']
num_parallel_calls = 4
num_concurrent_clients = 10  # should be greater than `num_parallel_calls`

bc_clients = [BertClient(port=5500, port_out=5501) for _ in range(num_concurrent_clients)]


def get_encodes(x, shuffle=False):
    # x is `batch_size` of lines, each of which is a json object
    samples = [json.loads(l) for l in x]
    texts = []
    for s in samples:
        t = s['fact']
        s_idx = random.randint(0, len(t) - 1) if shuffle else 0
        texts.append(t[s_idx: (s_idx + 40)])
    # get a client from available clients
    bc_client = bc_clients.pop()
    features = bc_client.encode(texts)
    # after use, put it back
    bc_clients.append(bc_client)
    return features


def get_ds(fp, batch_size=1024, shuffle=False, only_head=False):
    _get_encodes = lambda x: get_encodes(x, shuffle)
    ds = (tf.data.TextLineDataset(fp).batch(batch_size)
          .map(lambda x: tf.py_func(_get_encodes, [x], tf.float32, name='bert_client'),
               num_parallel_calls=num_parallel_calls))
    if shuffle:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10))
    if only_head:
        ds = ds.take(1).repeat(-1)
    return ds.prefetch(5).make_one_shot_iterator().get_next()


def get_config():
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement = False
    return config


quantizer = BaseQuantizer()
with tf.Session(config=get_config()) as sess:
    sess.run(tf.global_variables_initializer())
    iter, iter_per_save = 0, 100

    train_ds = get_ds(train_fp, shuffle=True)
    while True:
        x = sess.run(train_ds)
        loss, stat, _ = sess.run([quantizer.loss, quantizer.statistic, quantizer.train_op],
                                 feed_dict={quantizer.ph_x: x})
        iter += 1
        stat_str = ' '.join('%5s %.3f' % (k, v) for k, v in sorted(stat.items()))
        print('[T]%10d: %.5f %s' % (iter, loss, stat_str))
        if iter % iter_per_save == 0:
            dev_ds = get_ds(dev_fp)
            x = sess.run(dev_ds)
            loss, stat = sess.run([quantizer.loss, quantizer.statistic], feed_dict={quantizer.ph_x: x})
            stat_str = ' '.join('%5s %.3f' % (k, v) for k, v in sorted(stat.items()))
            print('[V]%10d: %.5f %s' % (iter, loss, stat_str))
