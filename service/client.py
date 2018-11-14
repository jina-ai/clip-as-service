#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import json
from datetime import datetime

import zmq


class BertClient:
    def __init__(self, ip='localhost', port=5555, output_fmt='ndarray'):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.identity = ('client-%d' % datetime.now().timestamp()).encode('ascii')
        self.socket.connect('tcp://%s:%d' % (ip, port))

        if output_fmt == 'ndarray':
            self.formatter = lambda x: x
        elif output_fmt == 'list':
            self.formatter = lambda x: x.tolist()
        else:
            raise AttributeError('"output_fmt" must be "ndarray" or "list"')

    def encode(self, texts):
        if self.is_valid_input(texts):
            self.socket.send_pyobj(texts)
            return self.formatter(self.socket.recv_pyobj())
        else:
            raise AttributeError('"texts" must be "List[str]"!')

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)

    @staticmethod
    def send_ndarray(socket, dest, X, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(X.dtype),
            shape=X.shape,
        )
        socket.send_multipart([dest, b'', json.dumps(md)], flags | zmq.SNDMORE)
        return socket.send_multipart([dest, b'', X], flags, copy=copy, track=track)
