#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import random
from datetime import datetime

import numpy as np
import zmq
from zmq.utils import jsonapi


class BertClient:
    def __init__(self, ip='localhost', port=5555, output_fmt='ndarray'):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.identity = ('client-%d' %
                                datetime.now().timestamp() + str(random.randint(0, 999))).encode('ascii')
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
            response = self.socket.recv_multipart()
            arr_info, arr_val = jsonapi.loads(response[0]), response[3]
            X = np.frombuffer(memoryview(arr_val), dtype=arr_info['dtype'])
            return self.formatter(X.reshape(arr_info['shape']))
        else:
            raise AttributeError('"texts" must be "List[str]"!')

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)
