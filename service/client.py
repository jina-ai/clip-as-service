#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import sys
import uuid

import numpy as np
import zmq
from zmq.utils import jsonapi

if sys.version_info >= (3, 0):
    _str = str
    _buffer = memoryview
else:
    # make it compatible for py2
    _str = basestring
    _buffer = buffer


class BertClient:
    def __init__(self, ip='localhost', port=5555, output_fmt='ndarray', show_server_config=True):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.identity = str(uuid.uuid4()).encode('ascii')
        self.socket.connect('tcp://%s:%d' % (ip, port))
        self.ip = ip
        self.port = port

        if output_fmt == 'ndarray':
            self.formatter = lambda x: x
        elif output_fmt == 'list':
            self.formatter = lambda x: x.tolist()
        else:
            raise AttributeError('"output_fmt" must be "ndarray" or "list"')

        if show_server_config:
            self.get_server_config()
            print('you should NOT see this message multiple times! '
                  'if you see it appears repeatedly, '
                  'please consider moving "BertClient()" out of the loop.')

    def get_server_config(self):
        self.socket.send(b'SHOW_CONFIG')
        response = self.socket.recv_multipart()
        print('the server at %s:%d returns the following config:' % (self.ip, self.port))
        for k, v in jsonapi.loads(response[0]).items():
            print('%30s\t=\t%-30s' % (k, v))

    def encode(self, texts):
        if self.is_valid_input(texts):
            self.socket.send_pyobj(texts)
            response = self.socket.recv_multipart()
            arr_info, arr_val = jsonapi.loads(response[0]), response[2]
            X = np.frombuffer(_buffer(arr_val), dtype=arr_info['dtype'])
            return self.formatter(X.reshape(arr_info['shape']))
        else:
            raise AttributeError('"texts" must be "List[str]" and non-empty!')

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, _str) and s.strip() for s in texts)
