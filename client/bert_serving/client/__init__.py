#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import sys
import threading
import time
import uuid
from collections import namedtuple

import numpy as np
import zmq
from zmq.utils import jsonapi

# in the future client version must match with server version
__version__ = '1.4.6'

if sys.version_info >= (3, 0):
    _str = str
    _buffer = memoryview
    _unicode = lambda x: x
else:
    # make it compatible for py2
    _str = basestring
    _buffer = buffer
    _unicode = lambda x: [BertClient.force_to_unicode(y) for y in x]

Response = namedtuple('Response', ['id', 'content'])


class BertClient:
    def __init__(self, ip='localhost', port=5555, port_out=5556,
                 output_fmt='ndarray', show_server_config=False,
                 identity=None, check_version=True):
        """ A client object connected to a BertServer

        Create a BertClient that connects with a BertServer
        Note, server must be ready at the moment.

        :param ip: the ip address of the server
        :param port: port for pushing data from client to server, must be consistent with the server side config
        :param port_out: port for publishing results from server to client, must be consistent with the server side config
        :param output_fmt: the output format of the sentence encodes,
        either in numpy array or python List[List[float]] (ndarray/list)
        :param show_server_config: whether to show server configs when first connected
        :param identity: the UUID of this client
        """
        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        self.identity = identity or str(uuid.uuid4()).encode('ascii')
        self.sender.connect('tcp://%s:%d' % (ip, port))

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.identity)
        self.receiver.connect('tcp://%s:%d' % (ip, port_out))

        self.request_id = 0
        self.pending_request = set()

        if output_fmt == 'ndarray':
            self.formatter = lambda x: x
        elif output_fmt == 'list':
            self.formatter = lambda x: x.tolist()
        else:
            raise AttributeError('"output_fmt" must be "ndarray" or "list"')

        self.output_fmt = output_fmt
        self.port = port
        self.port_out = port_out
        self.ip = ip

        s_status = self.server_status
        if check_version and s_status['server_version'] != self.status['client_version']:
            raise AttributeError('version mismatch! server version is %s but client version is %s!\n'
                                 'consider "pip install -U bert-serving-server bert-serving-client"' % (
                                     s_status['server_version'], self.status['client_version']))

        if show_server_config:
            self._print_dict(s_status, 'server config:')

        # print('you should NOT see this message multiple times! '
        #       'if you see it appears repeatedly, '
        #       'consider moving "BertClient()" out of the loop.')

    def _send(self, msg):
        self.sender.send_multipart([self.identity, msg, b'%d' % self.request_id])
        self.pending_request.add(self.request_id)
        self.request_id += 1

    def _recv(self):
        response = self.receiver.recv_multipart()
        request_id = int(response[-1])
        self.pending_request.remove(request_id)
        return Response(request_id, response)

    def _recv_ndarray(self):
        request_id, response = self._recv()
        arr_info, arr_val = jsonapi.loads(response[1]), response[2]
        X = np.frombuffer(_buffer(arr_val), dtype=arr_info['dtype'])
        return Response(request_id, self.formatter(X.reshape(arr_info['shape'])))

    @property
    def status(self):
        return {
            'identity': self.identity,
            'num_request': self.request_id,
            'num_pending_request': len(self.pending_request),
            'pending_request': self.pending_request,
            'output_fmt': self.output_fmt,
            'port': self.port,
            'port_out': self.port_out,
            'server_ip': self.ip,
            'client_version': __version__
        }

    @property
    def server_status(self):
        self._send(b'SHOW_CONFIG')
        return jsonapi.loads(self._recv().content[1])

    def encode(self, texts, blocking=True):
        """ Encode a list of strings to a list of vectors

        Note that if `blocking` is set to False, then you need to fetch the result manually afterwards.

        :param texts: list of sentence to be encoded. Larger list for better efficiency.
        :param blocking: wait until the encoded result is returned from the server. If false, will immediately return.
        :return: ndarray or a list[list[float]]
        """
        if self.is_valid_input(texts):
            texts = _unicode(texts)
            self._send(jsonapi.dumps(texts))
            return self._recv_ndarray().content if blocking else None
        else:
            raise AttributeError('"texts" must be "List[str]" and non-empty!')

    def fetch(self, delay=.0):
        """ Fetch the encoded vectors from server, use it with `encode(blocking=False)`

        Use it after `encode(texts, blocking=False)`. If there is no pending requests, will return None.
        Note that `fetch()` does not preserve the order of the requests! Say you have two non-blocking requests,
        R1 and R2, where R1 with 256 samples, R2 with 1 samples. It could be that R2 returns first.

        To fetch all results in the original sending order, please use `fetch_all(sort=True)`

        :param delay: delay in seconds and then run fetcher
        :return: tuple(int, ndarray), a generator that yields request id and encoded vector
        """
        time.sleep(delay)
        while self.pending_request:
            yield self._recv_ndarray()

    def fetch_all(self, sort=True, concat=False):
        """ Fetch all encoded vectors from server, use it with `encode(blocking=False)`

        Use it `encode(texts, blocking=False)`. If there is no pending requests, it will return None.

        :param sort: sort results by their request ids. It should be True if you want to preserve the sending order
        :param concat: concatenate all results into one ndarray
        :return: encoded vectors in a ndarray or a list of ndarray
        """
        if self.pending_request:
            tmp = list(self.fetch())
            if sort:
                tmp = sorted(tmp, key=lambda v: v.id)
            tmp = [v.content for v in tmp]
            if concat:
                if self.output_fmt == 'ndarray':
                    tmp = np.concatenate(tmp, axis=0)
                elif self.output_fmt == 'list':
                    tmp = [vv for v in tmp for vv in v]
            return tmp

    def encode_async(self, batch_generator, max_num_batch=None, delay=0.1):
        """ Async encode batches from a generator [Experimental, use with caution!]

        :param delay: delay in seconds and then run fetcher
        :param batch_generator: a generator that yields list[str] every time
        :param max_num_batch: stop after encoding this number of batches
        :return: a generator that yields encoded vectors in ndarray
        """

        def run():
            cnt = 0
            for texts in batch_generator:
                self.encode(texts, blocking=False)
                cnt += 1
                if max_num_batch and cnt == max_num_batch:
                    break

        t = threading.Thread(target=run)
        t.start()
        return self.fetch(delay)

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, _str) and s.strip() for s in texts)

    @staticmethod
    def force_to_unicode(text):
        return text if isinstance(text, unicode) else text.decode('utf-8')

    @staticmethod
    def _print_dict(x, title=None):
        if title:
            print(title)
        for k, v in x.items():
            print('%30s\t=\t%-30s' % (k, v))
