#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# read and write TFRecord

import os

import GPUtil
import tensorflow as tf
from bert_serving.client import BertClient

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
tf.logging.set_verbosity(tf.logging.INFO)

with open('README.md') as fp:
    data = [v for v in fp if v.strip()]
    bc = BertClient()
    list_vec = bc.encode(data)
    list_label = [0 for _ in data]  # a dummy list of all-zero labels

# write tfrecords

with tf.python_io.TFRecordWriter('tmp.tfrecord') as writer:
    def create_float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))


    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


    for (vec, label) in zip(list_vec, list_label):
        features = {'features': create_float_feature(vec), 'labels': create_int_feature([label])}
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

# read tfrecords and build dataset from it

num_hidden_unit = 768


def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    return tf.parse_single_example(record, {
        'features': tf.FixedLenFeature([num_hidden_unit], tf.float32),
        'labels': tf.FixedLenFeature([], tf.int64),
    })


ds = (tf.data.TFRecordDataset('tmp.tfrecord').repeat().shuffle(buffer_size=100).apply(
    tf.contrib.data.map_and_batch(lambda record: _decode_record(record), batch_size=64))
      .make_one_shot_iterator().get_next())

with tf.Session() as sess:
    while True:
        print(sess.run(ds))
