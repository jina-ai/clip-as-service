#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# using BertClient in sync way

import sys
import time

from service.client import BertClient

if __name__ == '__main__':
    bc = BertClient(port=int(sys.argv[1]), port_out=int(sys.argv[2]))
    # encode a list of strings
    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]

    for j in range(1, 200, 10):
        start_t = time.time()
        tmp = data * j
        bc.encode(tmp)
        time_t = time.time() - start_t
        print('encoding %d strs in %.2fs, speed: %d/s' % (len(tmp), time_t, int(len(tmp) / time_t)))
