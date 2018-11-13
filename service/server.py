#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import os
import pickle
import threading
import time
from math import ceil
from multiprocessing import Process

import GPUtil
import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator

from bert import tokenization, modeling
from bert.extract_features import model_fn_builder, convert_lst_to_features
from helper import set_logger

logger = set_logger()


class BertServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.model_dir = args.model_dir
        self.max_len = args.max_len
        self.num_worker = args.num_worker
        self.max_seq_per_worker = args.max_seq_per_worker
        self.port = args.port
        self.args = args

    def run(self):
        def get_a_worker():
            w = workers.pop(0)
            if not workers:
                # Don't poll clients if no workers are available
                poller.unregister(frontend)
            return w

        def free_a_worker(w):
            if not workers:
                # Poll for clients now that a worker is available
                poller.register(frontend, zmq.POLLIN)
            workers.append(w)

        context = zmq.Context.instance()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:%d' % self.port)
        backend = context.socket(zmq.ROUTER)
        backend.bind('ipc:///tmp/bert.service')

        available_gpus = GPUtil.getAvailable(limit=self.num_worker)
        if len(available_gpus) < self.num_worker:
            logger.warning('only %d GPU(s) is available, ask for %d' % (len(available_gpus), self.num_worker))
        for i in available_gpus:
            process = BertWorker(i, self.args)
            process.start()

        # Initialize main loop state
        workers = []
        poller = zmq.Poller()
        # Only poll for requests from backend until workers are available
        poller.register(backend, zmq.POLLIN)

        pending_part_jobs = {}
        finish_part_jobs = {}

        while True:
            logger.info('available workers: %d' % len(workers))
            sockets = dict(poller.poll())

            if backend in sockets:
                # Handle worker activity on the backend
                request = backend.recv_multipart()
                worker, _, client = request[:3]
                free_a_worker(worker)
                if client != b'READY' and len(request) > 3:
                    # If client reply, send rest back to frontend
                    _, reply = request[3:]
                    if client in pending_part_jobs and client in finish_part_jobs:
                        finish_part_jobs[client].extend(pickle.loads(reply))
                        # wait until all partial jobs from this client is done
                        # then concat all and send them back
                        if len(finish_part_jobs[client]) == pending_part_jobs[client]:
                            frontend.send_multipart([client, b'', pickle.dumps(finish_part_jobs[client])])
                            finish_part_jobs.pop(client)
                            pending_part_jobs.pop(client)
                    else:
                        frontend.send_multipart([client, b'', reply])

            if frontend in sockets:
                # Get next client request, route to last-used worker
                client, _, request = frontend.recv_multipart()
                seqs = pickle.loads(request)
                num_seqs = len(seqs)
                num_avail_worker = len(workers)

                if num_seqs > self.max_seq_per_worker and num_avail_worker > 1:
                    # divide the list by number of available workers
                    num_seq_each_worker = ceil(num_seqs / num_avail_worker)
                    s_idx = 0
                    pending_part_jobs[client] = num_seqs
                    finish_part_jobs[client] = []
                    while s_idx < num_seqs:
                        tmp = seqs[s_idx: (s_idx + num_seq_each_worker)]
                        if tmp:
                            worker = get_a_worker()
                            backend.send_multipart([worker, b'', client, b'', pickle.dumps(tmp)])
                        s_idx += len(tmp)
                else:
                    worker = get_a_worker()
                    backend.send_multipart([worker, b'', client, b'', request])

        frontend.close()
        backend.close()
        context.term()


class BertWorker(Process):
    def __init__(self, id, args):
        super().__init__()
        self.model_dir = args.model_dir
        self.config_fp = os.path.join(self.model_dir, 'bert_config.json')
        self.checkpoint_fp = os.path.join(self.model_dir, 'bert_model.ckpt')
        self.vocab_fp = os.path.join(args.model_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_fp)
        self.max_len = args.max_len
        self.worker_id = id
        self.daemon = True
        self.model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(self.config_fp),
            init_checkpoint=self.checkpoint_fp)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.worker_id)
        self.estimator = Estimator(self.model_fn)
        self.result = []

    def run(self):
        socket = zmq.Context().socket(zmq.REQ)
        socket.identity = u'worker-{}'.format(self.worker_id).encode('ascii')
        socket.connect('ipc:///tmp/bert.service')

        input_fn = self.input_fn_builder(socket)
        socket.send(b'READY')
        logger.info('worker %d is ready and listening' % self.worker_id)
        for r in self.estimator.predict(input_fn):
            self.result.append([round(float(x), 6) for x in r.flat])
        socket.close()
        logger.info('worker is terminated!')

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)

    def input_fn_builder(self, worker):
        def gen():
            while True:
                if self.result:
                    num_result = len(self.result)
                    worker.send_multipart([ident, b'', pickle.dumps(self.result)])
                    self.result.clear()
                    time_used = time.clock() - start
                    logger.info('encoded %d strs from %s in %.2fs @ %d/s' %
                                (num_result, ident, time_used, int(num_result / time_used)))
                ident, empty, msg = worker.recv_multipart()
                start = time.clock()
                msg = pickle.loads(msg)
                if self.is_valid_input(msg):
                    tmp_f = list(convert_lst_to_features(msg, self.max_len, self.tokenizer))
                    yield {
                        'input_ids': [f.input_ids for f in tmp_f],
                        'input_mask': [f.input_mask for f in tmp_f],
                        'input_type_ids': [f.input_type_ids for f in tmp_f]
                    }
                else:
                    logger.warning('worker %d: received unsupported type! sending back None' % self.id)
                    worker.send_multipart([ident, b'', pickle.dumps(None)])

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={k: tf.int32 for k in ['input_ids', 'input_mask', 'input_type_ids']},
                output_shapes={'input_ids': (None, self.max_len),
                               'input_mask': (None, self.max_len),
                               'input_type_ids': (None, self.max_len)}))

        return input_fn
