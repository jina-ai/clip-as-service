#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# using BertClient in async way

import sys

from service.client import BertClient


# an endless data stream, generating data in an extremely fast speed
def text_gen():
    while True:
        yield data


if __name__ == '__main__':
    bc = BertClient(port=int(sys.argv[1]), port_out=int(sys.argv[2]))

    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]

    # encoding without blocking:
    for _ in range(10):
        bc.encode(data, blocking=False)

    # then fetch all
    print(bc.fetch_all(concat=True))

    # get encoded vectors
    for j in bc.encode_async(text_gen(), max_num_batch=20):
        print('received %d : %s' % (j.id, j.content))
