#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# simple similarity search on FAQ

# from MulticoreTSNE import MulticoreTSNE as TSNE

with open('README.md') as fp:
    data = [v for v in fp if v.strip() and v.startswith('#####')]
