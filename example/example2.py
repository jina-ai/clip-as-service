#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# using BertClient in async way

import sys

from bert_serving.client import BertClient


def send_without_block(bc, data, repeat=10):
    # encoding without blocking:
    print('sending all data without blocking...')
    for _ in range(repeat):
        bc.encode(data, blocking=False)
    print('all sent!')


if __name__ == '__main__':
    bc = BertClient(port=int(sys.argv[1]), port_out=int(sys.argv[2]))
    num_repeat = 20

    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]

    send_without_block(bc, data, num_repeat)

    num_expect_vecs = len(data) * num_repeat

    # then fetch all
    print('now waiting until all results are available...')
    vecs = bc.fetch_all(concat=True)
    print('received %s, expected: %d' % (vecs.shape, num_expect_vecs))

    # now send it again
    send_without_block(bc, data, num_repeat)

    # this time fetch them one by one, due to the async encoding and server scheduling
    # sending order is NOT preserved!
    for v in bc.fetch():
        print('received %s, shape %s' % (v.id, v.content.shape))


    # finally let's do encode-fetch at the same time but in async mode
    # we do that by building an endless data stream, generating data in an extremely fast speed
    def text_gen():
        while True:
            yield data


    for j in bc.encode_async(text_gen(), max_num_batch=20):
        print('received %d : %s' % (j.id, j.content))
