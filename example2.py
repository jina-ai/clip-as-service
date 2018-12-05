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


def send_without_block(bc, data, repeat=10):
    # encoding without blocking:
    print('sending all data without blocking...')
    for _ in range(repeat):
        bc.encode(data, blocking=False)
    print('done!')


if __name__ == '__main__':
    bc = BertClient(port=int(sys.argv[1]), port_out=int(sys.argv[2]))

    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]

    send_without_block(bc, data, 10)

    num_expect_vecs = len(data) * 10

    # then fetch all
    print('now waiting until all results are available...')
    vecs = bc.fetch_all(concat=True)
    print('received %s, expected: %d' % (vecs.shape, num_expect_vecs))

    # now send it again
    send_without_block(bc, data, 10)

    # this time fetch them one by one, due to the async encoding and server scheduling
    # sending order is NOT preserved!
    for v in bc.fetch():
        print('received %s, shape %s' % (v.id, v.content.shape))

    # get encoded vectors
    for j in bc.encode_async(text_gen(), max_num_batch=20):
        print('received %d : %s' % (j.id, j.content))
