#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>


class BertClient:
    def __init__(self, ip='localhost', port=5555):
        import zmq
        from datetime import datetime
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.identity = ('client-%d' % datetime.now().timestamp()).encode('ascii')
        self.socket.connect('tcp://%s:%d' % (ip, port))

    def encode(self, texts):
        if self.is_valid_input(texts):
            self.socket.send_pyobj(texts)
            return self.socket.recv_pyobj()
        else:
            raise AttributeError('"texts" must be List[str]!')

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)
