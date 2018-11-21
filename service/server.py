#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import multiprocessing
import os
import pickle
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator
from zmq.utils import jsonapi

from bert import tokenization, modeling
from bert.extract_features import model_fn_builder, convert_lst_to_features
from helper import set_logger
from service.client import BertClient


class BertServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.model_dir = args.model_dir
        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.port = args.port
        self.args = args
        self.args_dict = {
            'model_dir': args.model_dir,
            'max_seq_len': args.max_seq_len,
            'num_worker': args.num_worker,
            'max_batch_size': args.max_batch_size,
            'port': args.port,
            'tensorflow_version': tf.__version__,
            'python_version': sys.version,
            'server_time': str(datetime.now())
        }
        self.processes = []
        self.frontend = None  # REQ->ROUTER
        self.backend = None  # PUSH->PULL
        self.context = None
        self.exit_flag = threading.Event()
        self.logger = set_logger('DISPATCHER')
        self.client_checksum = {}
        self.pending_client = {}
        self.pending_checksum = {}

    def close(self):
        self.logger.info('shutting down...')
        for p in self.processes:
            p.close()
        self.exit_flag.set()
        self.frontend.close()
        self.backend.close()
        self.context.term()
        self.logger.info('terminated!')

    def run(self):
        self.context = zmq.Context()
        self.frontend = self.context.socket(zmq.ROUTER)
        self.frontend.bind('tcp://*:%d' % self.port)
        # self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)

        self.backend = self.context.socket(zmq.PUSH)
        self.backend.bind('ipc://*')
        backend_addr = self.backend.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')

        # start the sink thread
        sink_thread = BertSink(self.args, self.frontend, self.client_checksum)
        sink_thread.start()
        self.processes.append(sink_thread)

        available_gpus = range(self.num_worker)
        try:
            import GPUtil
            available_gpus = GPUtil.getAvailable(limit=self.num_worker)
            if len(available_gpus) < self.num_worker:
                self.logger.warn('only %d GPU(s) is available, but ask for %d' % (len(available_gpus), self.num_worker))
        except FileNotFoundError:
            self.logger.warn('nvidia-smi is missing, often means no gpu found on this machine. '
                             'will run service on cpu instead')

        # start the backend processes
        for i in available_gpus:
            process = BertWorker(i, self.args, backend_addr, sink_thread.address)
            self.processes.append(process)
            process.start()

        while not self.exit_flag.is_set():
            client, _, msg = self.frontend.recv_multipart()
            if msg == b'SHOW_CONFIG':
                self.frontend.send_multipart(
                    [client, b'',
                     jsonapi.dumps({**{'client': client.decode('ascii'),
                                       'num_process': len(self.processes),
                                       'ipc_backend': backend_addr,
                                       'ipc_sink': sink_thread.address}, **self.args_dict})])
                continue

            seqs = pickle.loads(msg)
            num_seqs = len(seqs)
            self.client_checksum[client] = num_seqs

            if num_seqs > self.max_batch_size:
                # divide the large batch into small batches
                s_idx = 0
                while s_idx < num_seqs:
                    tmp = seqs[s_idx: (s_idx + self.max_batch_size)]
                    if tmp:
                        # get the worker with minimum workload
                        client_partial_id = client + b'@%d' % s_idx
                        self.backend.send_multipart([client_partial_id, b'', pickle.dumps(tmp, protocol=-1)])
                    s_idx += len(tmp)
            else:
                self.backend.send_multipart([client, b'', msg])

        self.frontend.close()
        self.backend.close()
        self.context.term()


class BertSink(threading.Thread):
    def __init__(self, args, frontend, client_chk):
        super().__init__()
        self.port = args.port
        self.context = None
        self.receiver = None
        self.frontend = frontend
        self.exit_flag = threading.Event()
        self.logger = set_logger('SINK')
        self.address = None
        self.client_checksum = client_chk

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.logger.info('terminated!')

    def run(self):
        self.context = zmq.Context()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind('ipc://*')
        self.address = self.receiver.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')
        pending_checksum = defaultdict(int)
        pending_client = defaultdict(list)

        while not self.exit_flag.is_set():
            msg = self.receiver.recv_multipart()
            client_id = msg[0]
            # parsing the ndarray
            arr_info, arr_val = jsonapi.loads(msg[2]), msg[4]
            X = np.frombuffer(memoryview(arr_val), dtype=arr_info['dtype'])
            X = X.reshape(arr_info['shape'])
            client_info = client_id.split(b'@')
            client_id = client_info[0]
            partial_id = client_info[1] if len(client_info) == 2 else 0
            pending_client[client_id].append((X, partial_id))
            pending_checksum[client_id] += X.shape[0]
            self.logger.info('received %d of client %s (%d/%d)' % (X.shape[0], client_id,
                                                                   pending_checksum[client_id],
                                                                   self.client_checksum[client_id]))

            # check if there are finished jobs, send it back to workers
            finished = [(k, v) for k, v in pending_client.items() if pending_checksum[k] == self.client_checksum[k]]
            for client, tmp in finished:
                self.logger.info(
                    'client %s %d samples are done! sending back to client' % (client, self.client_checksum[client]))
                # re-sort to the original order
                tmp = [x[0] for x in sorted(tmp, key=lambda x: x[1])]
                send_ndarray(self.frontend, client, np.concatenate(tmp, axis=0))
                pending_client.pop(client)
                pending_checksum.pop(client)
                self.client_checksum.pop(client)

        self.receiver.close()
        self.context.term()


class BertWorker(Process):
    def __init__(self, id, args, worker_address, sink_address):
        super().__init__()
        self.model_dir = args.model_dir
        self.config_fp = os.path.join(self.model_dir, 'bert_config.json')
        self.checkpoint_fp = os.path.join(self.model_dir, 'bert_model.ckpt')
        self.vocab_fp = os.path.join(args.model_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_fp)
        self.max_seq_len = args.max_seq_len
        self.worker_id = id
        self.daemon = True
        self.model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(self.config_fp),
            init_checkpoint=self.checkpoint_fp,
            pooling_strategy=args.pooling_strategy,
            pooling_layer=args.pooling_layer
        )
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.worker_id)
        self.estimator = Estimator(self.model_fn)
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger('WORKER-%d' % self.worker_id)
        self.worker_address = worker_address
        self.sink_address = sink_address

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.connect(self.worker_address)

        sink = context.socket(zmq.PUSH)
        sink.connect(self.sink_address)

        input_fn = self.input_fn_builder(receiver)

        self.logger.info('ready and listening')
        start_t = time.perf_counter()
        for r in self.estimator.predict(input_fn, yield_single_examples=False):
            # logger.info('new result!')
            send_ndarray(sink, r['client_id'], r['encodes'])
            time_used = time.perf_counter() - start_t
            start_t = time.perf_counter()
            self.logger.info('job %s\tsamples: %4d\tdone: %.2fs' %
                             (r['client_id'], r['encodes'].shape[0], time_used))

        receiver.close()
        sink.close()
        context.term()

    def input_fn_builder(self, worker):
        def gen():
            while not self.exit_flag.is_set():
                client_id, empty, msg = worker.recv_multipart()
                msg = pickle.loads(msg)
                self.logger.info('received %4d from %s' % (len(msg), client_id))
                if BertClient.is_valid_input(msg):
                    tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, self.tokenizer))
                    yield {
                        'client_id': client_id,
                        'input_ids': [f.input_ids for f in tmp_f],
                        'input_mask': [f.input_mask for f in tmp_f],
                        'input_type_ids': [f.input_type_ids for f in tmp_f]
                    }
                else:
                    self.logger.warning('received unsupported type from %s! sending back None' % client_id)
                    worker.send_multipart([client_id, b'', b''])
            worker.close()

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32,
                              'client_id': tf.string},
                output_shapes={
                    'client_id': (),
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len),
                    'input_type_ids': (None, self.max_seq_len)}))

        return input_fn


def send_ndarray(src, dest, X, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype=str(X.dtype), shape=X.shape)
    return src.send_multipart([dest, b'', jsonapi.dumps(md), b'', X],
                              flags, copy=copy, track=track)
