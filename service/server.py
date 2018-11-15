#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import multiprocessing
import os
import pickle
import sys
import threading
import time
from datetime import datetime
from multiprocessing import Process

import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator
from zmq.utils import jsonapi

from bert import tokenization, modeling
from bert.extract_features import model_fn_builder, convert_lst_to_features
from helper import set_logger
from service.client import BertClient

logger = set_logger()
WORKER_ADDR = 'ipc:///tmp/bert.workers'
SINK_ADDR = 'ipc:///tmp/bert.sink'


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
        self.sink = None  # PUSH->PULL
        self.context = None

    def close(self):
        logger.info('shutting down bert-server...')
        for p in self.processes:
            p.close()
        self.frontend.close()
        self.backend.close()
        self.sink.close()
        self.context.term()
        self.join()
        logger.info('bert-server is terminated!')

    def run(self):
        self.context = zmq.Context()
        self.frontend = self.context.socket(zmq.ROUTER)
        self.frontend.bind('tcp://*:%d' % self.port)

        self.backend = self.context.socket(zmq.PUSH)
        self.backend.bind(WORKER_ADDR)

        self.sink = self.context.socket(zmq.PUSH)
        self.sink.bind(SINK_ADDR)

        available_gpus = range(self.num_worker)
        try:
            import GPUtil
            available_gpus = GPUtil.getAvailable(limit=self.num_worker)
            if len(available_gpus) < self.num_worker:
                logger.warning('only %d GPU(s) is available, but ask for %d' % (len(available_gpus), self.num_worker))
        except FileNotFoundError:
            logger.warn('nvidia-smi is missing, often means no gpu found on this machine. '
                        'will run service on cpu instead')

        for i in available_gpus:
            process = BertWorker(i, self.args)
            self.processes.append(process)
            process.start()

        while True:
            client, _, msg = self.frontend.recv_multipart()
            if msg == b'SHOW_CONFIG':
                self.frontend.send_multipart(
                    [client, b'',
                     jsonapi.dumps({**{'client': client.decode('ascii')}, **self.args_dict})])
                continue

            seqs = pickle.loads(msg)
            num_seqs = len(seqs)

            if num_seqs > self.max_batch_size:
                # divide the large batch into small batches
                s_idx = 0
                while s_idx < num_seqs:
                    tmp = seqs[s_idx: (s_idx + self.max_batch_size)]
                    if tmp:
                        # get the worker with minimum workload
                        self.backend.send_multipart([client, b'', pickle.dumps(tmp, protocol=-1)])
                    s_idx += len(tmp)
            else:
                self.backend.send_multipart([client, b'', msg])


class BertWorker(Process):
    def __init__(self, id, args):
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
            init_checkpoint=self.checkpoint_fp)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.worker_id)
        self.estimator = Estimator(self.model_fn)

        self.context = None
        self.receiver = None
        self.sink = None

        self.exit_flag = multiprocessing.Event()

    def close(self):
        logger.info('shutting down bert-worker %d ...' % self.worker_id)
        self.exit_flag.set()
        self.receiver.close()
        self.sink.close()
        self.context.term()
        self.terminate()
        self.join()
        logger.info('bert-worker %d is terminated!' % self.worker_id)

    def run(self):
        self.context = zmq.Context()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.connect(WORKER_ADDR)

        self.sink = self.context.socket(zmq.PUSH)
        self.sink.connect(SINK_ADDR)

        input_fn = self.input_fn_builder(self.receiver)

        logger.info('worker %d is ready and listening' % self.worker_id)
        start_t = time.perf_counter()
        for r in self.estimator.predict(input_fn, yield_single_examples=False):
            send_ndarray(self.sink, r['client_id'], r['encodes'])
            time_used = time.perf_counter() - start_t
            start_t = time.perf_counter()
            logger.info('bert-worker %2d: job %s\tsamples: %4d\tdone: %.2fs' %
                        (self.worker_id, r['client_id'], r['encodes'].shape[0], time_used))

    def input_fn_builder(self, worker):
        def gen():
            while not self.exit_flag.is_set():
                client_id, empty, msg = worker.recv_multipart()
                msg = pickle.loads(msg)
                logger.info('bert-worker %2d received %4d from %s' % (self.worker_id, len(msg), client_id))
                if BertClient.is_valid_input(msg):
                    tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, self.tokenizer))
                    yield {
                        'client_id': client_id,
                        'input_ids': [f.input_ids for f in tmp_f],
                        'input_mask': [f.input_mask for f in tmp_f],
                        'input_type_ids': [f.input_type_ids for f in tmp_f]
                    }
                else:
                    logger.warning('received unsupported type from %s! sending back None' % client_id)
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
    md = dict(
        dtype=str(X.dtype),
        shape=X.shape,
    )
    src.send_multipart([dest, b'', jsonapi.dumps(md)], flags | zmq.SNDMORE)
    return src.send_multipart([dest, b'', X], flags, copy=copy, track=track)
