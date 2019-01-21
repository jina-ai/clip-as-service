#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import time

import GPUtil
from bert_serving.server.helper import get_run_args
from termcolor import colored

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
from bert_serving.client import BertClient


class TimeContext:
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        self.start = time.perf_counter()
        print(self._msg, end=' ...\t', flush=True)

    def __exit__(self, typ, value, traceback):
        self.duration = time.perf_counter() - self.start
        print(colored('    [%3.0f secs]' % self.duration, 'green'), flush=True)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_file', type=str, required=True,
                        help='file name of the input text')
    parser.add_argument('-out_file', type=str, required=True,
                        help='file name of the output tfrecords')
    parser.add_argument('-ip', type=str, default='localhost',
                        help='ip address of the bert server')
    parser.add_argument('-port', type=int, default=5555,
                        help='port of the bert server')
    parser.add_argument('-port_out', type=int, default=5556,
                        help='output port of the bert server')
    parser.add_argument('-batch_size', type=int, default=4096)
    parser.add_argument('-max_num_line', type=int, default=1000000)
    return parser


def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def run(args):
    def encode_write():
        with TimeContext('encoded %d lines' % (num_examples + len(buffer))):
            for vec in bc.encode(buffer):
                features = {'features': create_float_feature(vec)}
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
            buffer.clear()

    num_examples = 0
    with open(args.in_file) as fp, tf.python_io.TFRecordWriter(args.out_file) as writer:
        bc = BertClient(args.ip, args.port, args.port_out)
        buffer = []
        for v in fp:
            if v.strip():
                buffer.append(v.strip())

            if len(buffer) > args.batch_size:
                encode_write()
                num_examples += len(buffer)

            if num_examples >= args.max_num_line:
                break

        if buffer and num_examples + len(buffer) < args.max_num_line:
            encode_write()


if __name__ == '__main__':
    args = get_run_args(get_args_parser)
    run(args)
