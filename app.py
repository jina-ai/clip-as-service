#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import argparse
import sys

try:
    import gpu_env
except:
    print('no GPUutils!')
from service.server import BertServer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', type=str, default='/data/cips/save/chinese-bert/chinese_L-12_H-768_A-12/',
                        help='pretrained BERT model')
    parser.add_argument('-max_len', type=int, default=25,
                        help='maximum length of a sequence')
    parser.add_argument('-num_worker', type=int, default=2,
                        help='number of server instances')
    parser.add_argument('-port', type=int, default=5555,
                        help='port number for C-S communication')

    args = parser.parse_args()
    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage:\n{0}\nparameters: \n{1}'.format(' '.join([x for x in sys.argv]), param_str))
    return args


if __name__ == '__main__':
    args = get_args()
    server = BertServer(args)
    server.start()
    server.join()
