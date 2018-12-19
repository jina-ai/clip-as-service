#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# simple similarity search on FAQ

import scipy.spatial.distance
from bert_serving.client import BertClient

prefix_q = '##### **Q:** '

# start your server, e.g.

with open('README.md') as fp, BertClient() as bc:
    docs = [v.replace(prefix_q, '') for v in fp if v.strip() and v.startswith(prefix_q)]
    doc_vecs = bc.encode(docs)

    while True:
        query = input('your question: ')
        query_vec = bc.encode([query])
        score = scipy.spatial.distance.cdist(doc_vecs, query_vec, 'cosine')
        print(score)
