#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import sys
import threading
import time
import uuid
import warnings
from collections import namedtuple

import numpy as np
import zmq
from zmq.utils import jsonapi

__all__ = ['__version__', 'BertClient']

# in the future client version must match with server version
__version__ = '1.7.1'

if sys.version_info >= (3, 0):
    _py2 = False
    _str = str
    _buffer = memoryview
    _unicode = lambda x: x
else:
    # make it compatible for py2
    _py2 = True
    _str = basestring
    _buffer = buffer
    _unicode = lambda x: [BertClient._force_to_unicode(y) for y in x]

Response = namedtuple('Response', ['id', 'content'])


class BertClient:
    def __init__(self, ip='localhost', port=5555, port_out=5556,
                 output_fmt='ndarray', show_server_config=False,
                 identity=None, check_version=True, check_length=True,
                 timeout=-1):
        """ A client object connected to a BertServer

        Create a BertClient that connects to a BertServer.
        Note, server must be ready at the moment you are calling this function.
        If you are not sure whether the server is ready, then please set `check_version=False` and `check_length=False`

        You can also use it as a context manager:

        .. highlight:: python
        .. code-block:: python

            with BertClient() as bc:
                bc.encode(...)

            # bc is automatically closed out of the context

        :type timeout: int
        :type check_version: bool
        :type check_length: bool
        :type identity: str
        :type show_server_config: bool
        :type output_fmt: str
        :type port_out: int
        :type port: int
        :type ip: str
        :param ip: the ip address of the server
        :param port: port for pushing data from client to server, must be consistent with the server side config
        :param port_out: port for publishing results from server to client, must be consistent with the server side config
        :param output_fmt: the output format of the sentence encodes, either in numpy array or python List[List[float]] (ndarray/list)
        :param show_server_config: whether to show server configs when first connected
        :param identity: the UUID of this client
        :param check_version: check if server has the same version as client, raise AttributeError if not the same
        :param check_length: check if server `max_seq_len` is less than the sentence length before sent
        :param timeout: set the timeout (milliseconds) for receive operation on the client, -1 means no timeout and wait until result returns
        """

        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.identity = identity or str(uuid.uuid4()).encode('ascii')
        self.sender.connect('tcp://%s:%d' % (ip, port))

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.identity)
        self.receiver.connect('tcp://%s:%d' % (ip, port_out))

        self.request_id = 0
        self.timeout = timeout
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
        self.length_limit = 0

        if check_version or show_server_config or check_length:
            s_status = self.server_status

            if check_version and s_status['server_version'] != self.status['client_version']:
                raise AttributeError('version mismatch! server version is %s but client version is %s!\n'
                                     'consider "pip install -U bert-serving-server bert-serving-client"\n'
                                     'or disable version-check by "BertClient(check_version=False)"' % (
                                         s_status['server_version'], self.status['client_version']))

            if show_server_config:
                self._print_dict(s_status, 'server config:')

            if check_length:
                self.length_limit = int(s_status['max_seq_len'])

    def close(self):
        """
            Gently close all connections of the client. If you are using BertClient as context manager,
            then this is not necessary.

        """
        self.sender.close()
        self.receiver.close()
        self.context.term()

    def _send(self, msg, msg_len=0):
        self.sender.send_multipart([self.identity, msg, b'%d' % self.request_id, b'%d' % msg_len])
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
        X = np.frombuffer(_buffer(arr_val), dtype=str(arr_info['dtype']))
        return Response(request_id, self.formatter(X.reshape(arr_info['shape'])))

    @property
    def status(self):
        """
            Get the status of this BertClient instance

        :rtype: dict[str, str]
        :return: a dictionary contains the status of this BertClient instance

        """
        return {
            'identity': self.identity,
            'num_request': self.request_id,
            'num_pending_request': len(self.pending_request),
            'pending_request': self.pending_request,
            'output_fmt': self.output_fmt,
            'port': self.port,
            'port_out': self.port_out,
            'server_ip': self.ip,
            'client_version': __version__,
            'timeout': self.timeout
        }

    def _timeout(func):
        def arg_wrapper(self, *args, **kwargs):
            if 'blocking' in kwargs and not kwargs['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, *args, **kwargs)
            except zmq.error.Again as _e:
                t_e = TimeoutError(
                    'no response from the server (with "timeout"=%d ms), please check the following:'
                    'is the server still online? is the network broken? are "port" and "port_out" correct? '
                    'are you encoding a huge amount of data whereas the timeout is too small for that?' % self.timeout)
                if _py2:
                    raise t_e
                else:
                    raise t_e from _e
            finally:
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)

        return arg_wrapper

    @property
    @_timeout
    def server_status(self):
        """
            Get the current status of the server connected to this client

        :return: a dictionary contains the current status of the server connected to this client
        :rtype: dict[str, str]

        """
        self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
        self._send(b'SHOW_CONFIG')
        return jsonapi.loads(self._recv().content[1])

    @_timeout
    def encode(self, texts, blocking=True, is_tokenized=False):
        """ Encode a list of strings to a list of vectors

        `texts` should be a list of strings, each of which represents a sentence.
        If `is_tokenized` is set to True, then `texts` should be list[list[str]],
        outer list represents sentence and inner list represent tokens in the sentence.
        Note that if `blocking` is set to False, then you need to fetch the result manually afterwards.

        .. highlight:: python
        .. code-block:: python

            with BertClient() as bc:
                # encode untokenized sentences
                bc.encode(['First do it',
                          'then do it right',
                          'then do it better'])

                # encode tokenized sentences
                bc.encode([['First', 'do', 'it'],
                           ['then', 'do', 'it', 'right'],
                           ['then', 'do', 'it', 'better']], is_tokenized=True)

        :type is_tokenized: bool
        :type blocking: bool
        :type timeout: bool
        :type texts: list[str] or list[list[str]]
        :param is_tokenized: whether the input texts is already tokenized
        :param texts: list of sentence to be encoded. Larger list for better efficiency.
        :param blocking: wait until the encoded result is returned from the server. If false, will immediately return.
        :param timeout: throw a timeout error when the encoding takes longer than the predefined timeout.
        :return: encoded sentence/token-level embeddings, rows correspond to sentences
        :rtype: numpy.ndarray or list[list[float]]

        """
        if is_tokenized:
            self._check_input_lst_lst_str(texts)
        else:
            self._check_input_lst_str(texts)

        if self.length_limit and not self._check_length(texts, self.length_limit, is_tokenized):
            warnings.warn('some of your sentences have more tokens than "max_seq_len=%d" set on the server, '
                          'as consequence you may get less-accurate or truncated embeddings.\n'
                          'here is what you can do:\n'
                          '- disable the length-check by create a new "BertClient(check_length=False)" '
                          'when you do not want to display this warning\n'
                          '- or, start a new server with a larger "max_seq_len"' % self.length_limit)

        texts = _unicode(texts)
        self._send(jsonapi.dumps(texts), len(texts))
        return self._recv_ndarray().content if blocking else None

    def fetch(self, delay=.0):
        """ Fetch the encoded vectors from server, use it with `encode(blocking=False)`

        Use it after `encode(texts, blocking=False)`. If there is no pending requests, will return None.
        Note that `fetch()` does not preserve the order of the requests! Say you have two non-blocking requests,
        R1 and R2, where R1 with 256 samples, R2 with 1 samples. It could be that R2 returns first.

        To fetch all results in the original sending order, please use `fetch_all(sort=True)`

        :type delay: float
        :param delay: delay in seconds and then run fetcher
        :return: a generator that yields request id and encoded vector in a tuple, where the request id can be used to determine the order
        :rtype: Iterator[tuple(int, numpy.ndarray)]

        """
        time.sleep(delay)
        while self.pending_request:
            yield self._recv_ndarray()

    def fetch_all(self, sort=True, concat=False):
        """ Fetch all encoded vectors from server, use it with `encode(blocking=False)`

        Use it `encode(texts, blocking=False)`. If there is no pending requests, it will return None.

        :type sort: bool
        :type concat: bool
        :param sort: sort results by their request ids. It should be True if you want to preserve the sending order
        :param concat: concatenate all results into one ndarray
        :return: encoded sentence/token-level embeddings in sending order
        :rtype: numpy.ndarray or list[list[float]]

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

    def encode_async(self, batch_generator, max_num_batch=None, delay=0.1, is_tokenized=False):
        """ Async encode batches from a generator

        :param is_tokenized: whether batch_generator generates tokenized sentences
        :param delay: delay in seconds and then run fetcher
        :param batch_generator: a generator that yields list[str] or list[list[str]] (for `is_tokenized=True`) every time
        :param max_num_batch: stop after encoding this number of batches
        :return: a generator that yields encoded vectors in ndarray, where the request id can be used to determine the order
        :rtype: Iterator[tuple(int, numpy.ndarray)]

        """

        def run():
            cnt = 0
            for texts in batch_generator:
                self.encode(texts, blocking=False, is_tokenized=is_tokenized)
                cnt += 1
                if max_num_batch and cnt == max_num_batch:
                    break

        t = threading.Thread(target=run)
        t.start()
        return self.fetch(delay)

    @staticmethod
    def _check_length(texts, len_limit, tokenized):
        if tokenized:
            # texts is already tokenized as list of str
            return all(len(t) <= len_limit for t in texts)
        else:
            # do a simple whitespace tokenizer
            return all(len(t.split()) <= len_limit for t in texts)

    @staticmethod
    def _check_input_lst_str(texts):
        if not isinstance(texts, list):
            raise TypeError('"%s" must be %s, but received %s' % (texts, type([]), type(texts)))
        if not len(texts):
            raise ValueError(
                '"%s" must be a non-empty list, but received %s with %d elements' % (texts, type(texts), len(texts)))
        for idx, s in enumerate(texts):
            if not isinstance(s, _str):
                raise TypeError('all elements in the list must be %s, but element %d is %s' % (type(''), idx, type(s)))
            if not s.strip():
                raise ValueError(
                    'all elements in the list must be non-empty string, but element %d is %s' % (idx, repr(s)))

    @staticmethod
    def _check_input_lst_lst_str(texts):
        if not isinstance(texts, list):
            raise TypeError('"texts" must be %s, but received %s' % (type([]), type(texts)))
        if not len(texts):
            raise ValueError(
                '"texts" must be a non-empty list, but received %s with %d elements' % (type(texts), len(texts)))
        for s in texts:
            BertClient._check_input_lst_str(s)

    @staticmethod
    def _force_to_unicode(text):
        return text if isinstance(text, unicode) else text.decode('utf-8')

    @staticmethod
    def _print_dict(x, title=None):
        if title:
            print(title)
        for k, v in x.items():
            print('%30s\t=\t%-30s' % (k, v))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
