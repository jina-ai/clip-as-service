#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import multiprocessing
import os
import pickle
import threading
import time
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator

from bert import tokenization, modeling
from bert.extract_features import model_fn_builder, convert_lst_to_features
from helper import set_logger
from service.client import BertClient

logger = set_logger()


class BertServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.model_dir = args.model_dir
        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.port = args.port
        self.args = args
        self.processes, self.workers = [], []
        self.frontend, self.backend, self.context = None, None, None

    def close(self):
        logger.info('shutting down bert-server...')
        for p in self.processes:
            p.close()
        self.frontend.close()
        self.backend.close()
        self.context.term()
        self.join()
        logger.info('bert-server is terminated!')

    def run(self):
        def get_a_worker():
            return self.workers.pop(0)

        def free_a_worker(w):
            self.workers.append(w)

        def register_job(c, num_part=1):
            job_checksum[c] = num_part
            finish_jobs[c] = []

        def unregister_job(c):
            job_checksum.pop(c)
            finish_jobs.pop(c)

        self.context = zmq.Context.instance()
        self.frontend = self.context.socket(zmq.ROUTER)
        self.frontend.bind('tcp://*:%d' % self.port)
        self.backend = self.context.socket(zmq.ROUTER)
        self.backend.bind('ipc:///tmp/bert.service')

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

        poller = zmq.Poller()
        # Only poll for requests from backend until workers are available
        poller.register(self.backend, zmq.POLLIN)

        job_queue, finish_jobs, job_checksum = [], {}, {}

        while True:
            sockets = dict(poller.poll(2))

            if self.backend in sockets:
                md = self.backend.recv_json()
                request = self.backend.recv_multipart()
                worker, _, client = request[:3]
                free_a_worker(worker)
                if client != b'READY' and len(request) > 3:
                    _, reply = request[3:]
                    X = np.frombuffer(memoryview(reply), dtype=md['dtype'])
                    finish_jobs[client].append(X.reshape(md['shape']))
                else:
                    poller.register(self.frontend, zmq.POLLIN)

            if self.frontend in sockets:
                # Get next client request, route to last-used worker
                client, _, request = self.frontend.recv_multipart()
                seqs = pickle.loads(request)
                num_seqs = len(seqs)

                if num_seqs > self.max_batch_size:
                    # divide the large batch into small batches
                    s_idx = 0
                    n = 0
                    while s_idx < num_seqs:
                        tmp = seqs[s_idx: (s_idx + self.max_batch_size)]
                        if tmp:
                            job_queue.append((client, pickle.dumps(tmp, protocol=-1)))
                            n += 1
                        s_idx += len(tmp)
                    register_job(client, num_part=n)
                else:
                    register_job(client)
                    job_queue.append((client, request))

            # check if there are finished jobs, send it back to workers
            finished = [(k, v) for k, v in finish_jobs.items() if len(v) == job_checksum[k]]
            for client, tmp in finished:
                self.frontend.send_multipart([client, b'', pickle.dumps(np.concatenate(tmp, axis=0), protocol=-1)])
                unregister_job(client)

            # non-empty job queue and free workers, pop the last one and send it to a worker
            while self.workers and job_queue:
                client, tmp = job_queue.pop()
                worker = get_a_worker()
                self.backend.send_multipart([worker, b'', client, b'', tmp])
                logger.info('available workers: %2d\tjob queue: %3d\tpending clients: %3d' % (
                    len(self.workers), len(job_queue), len(job_checksum)))


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
        self.dest = None
        self._start_t = time.perf_counter()
        self.socket = None
        self.exit_flag = multiprocessing.Event()

    def close(self):
        logger.info('shutting down bert-worker %d ...' % self.worker_id)
        self.exit_flag.set()
        self.terminate()
        self.join()
        logger.info('bert-worker %d is terminated!' % self.worker_id)

    def run(self):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.identity = u'worker-{}'.format(self.worker_id).encode('ascii')
        self.socket.connect('ipc:///tmp/bert.service')

        input_fn = self.input_fn_builder(self.socket)
        self.socket.send(b'READY')
        logger.info('worker %d is ready and listening' % self.worker_id)
        for r in self.estimator.predict(input_fn, yield_single_examples=False):
            self.socket.send_multipart([self.dest, b'', pickle.dumps(r, protocol=-1)])
            time_used = time.perf_counter() - self._start_t
            logger.info('job %s is done in %.2fs' % (self.dest, time_used))

    def input_fn_builder(self, worker):
        def gen():
            while not self.exit_flag.is_set():
                self.dest, empty, msg = worker.recv_multipart()
                self._start_t = time.perf_counter()
                msg = pickle.loads(msg)
                if BertClient.is_valid_input(msg):
                    tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, self.tokenizer))
                    yield {
                        'input_ids': [f.input_ids for f in tmp_f],
                        'input_mask': [f.input_mask for f in tmp_f],
                        'input_type_ids': [f.input_type_ids for f in tmp_f]
                    }
                else:
                    logger.warning('worker %s: received unsupported type! sending empty back' % self.dest)
                    worker.send_multipart([self.dest, b'', b''])
            self.socket.close()

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={k: tf.int32 for k in ['input_ids', 'input_mask', 'input_type_ids']},
                output_shapes={'input_ids': (None, self.max_seq_len),
                               'input_mask': (None, self.max_seq_len),
                               'input_type_ids': (None, self.max_seq_len)}))

        return input_fn
