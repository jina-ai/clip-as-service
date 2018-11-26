#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import multiprocessing
import os
import sys
import threading
import time
import uuid
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
        self.logger = set_logger('VENTILATOR')

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
        self.context = zmq.Context()

        # frontend facing client
        self.frontend = self.context.socket(zmq.PULL)
        self.frontend.bind('tcp://*:%d' % self.port)

        # pair connection between frontend and sink
        self.sink = self.context.socket(zmq.PAIR)
        self.sink.bind('ipc://*')

        # backend facing workers
        self.backend = self.context.socket(zmq.PUSH)
        self.backend.bind('ipc://*')
        self.addr_backend = self.backend.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')

        # start the sink thread
        proc_sink = BertSink(self.args, self.sink.getsockopt(zmq.LAST_ENDPOINT).decode('ascii'))
        proc_sink.start()
        self.processes.append(proc_sink)
        self.addr_sink = self.sink.recv().decode('ascii')
        self.logger.info('frontend-sink ipc: %s' % self.addr_sink)

    def close(self):
        self.logger.info('shutting down...')
        for p in self.processes:
            p.close()
        self.frontend.close()
        self.backend.close()
        self.sink.close()
        self.context.term()
        self.logger.info('terminated!')

    def run(self):
        available_gpus = range(self.num_worker)
        try:
            import GPUtil
            available_gpus = GPUtil.getAvailable(limit=self.num_worker, maxLoad=0.1, maxMemory=0.01)
            if len(available_gpus) < self.num_worker:
                self.logger.warn('only %d GPU(s) is available, but ask for %d' % (len(available_gpus), self.num_worker))
        except FileNotFoundError:
            self.logger.warn('nvidia-smi is missing, often means no gpu found on this machine. '
                             'will run service on cpu instead')

        # start the backend processes
        for i in available_gpus:
            process = BertWorker(i, self.args, self.addr_backend, self.addr_sink)
            self.processes.append(process)
            process.start()

        try:
            while True:
                client, msg = self.frontend.recv_multipart()
                client = client + b'#' + str(uuid.uuid4()).encode('ascii')
                seqs = jsonapi.loads(msg)
                num_seqs = len(seqs)
                # tell sink to collect a new job
                self.sink.send_multipart([client, b'%d' % num_seqs])

                if num_seqs > self.max_batch_size:
                    # divide the large batch into small batches
                    s_idx = 0
                    while s_idx < num_seqs:
                        tmp = seqs[s_idx: (s_idx + self.max_batch_size)]
                        if tmp:
                            # get the worker with minimum workload
                            client_partial_id = client + b'@%d' % s_idx
                            self.backend.send_multipart([client_partial_id, jsonapi.dumps(tmp)])
                        s_idx += len(tmp)
                else:
                    self.backend.send_multipart([client, msg])
        except zmq.error.ContextTerminated:
            self.logger.error('context is closed!')


class BertSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger('SINK')
        self.front_sink_addr = front_sink_addr

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.logger.info('terminated!')

    def run(self):
        context = zmq.Context()
        # receive from workers
        receiver = context.socket(zmq.PULL)
        receiver.bind('ipc://*')

        frontend = context.socket(zmq.PAIR)
        frontend.connect(self.front_sink_addr)

        # publish to client
        sender = context.socket(zmq.PUB)
        sender.bind('tcp://*:%d' % self.port)

        pending_checksum = defaultdict(int)
        pending_result = defaultdict(list)
        job_checksum = {}

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver.getsockopt(zmq.LAST_ENDPOINT))

        try:
            while True:
                socks = dict(poller.poll())
                if socks.get(receiver) == zmq.POLLIN:
                    msg = receiver.recv_multipart()
                    job_id = msg[0]
                    # parsing the ndarray
                    arr_info, arr_val = jsonapi.loads(msg[1]), msg[2]
                    X = np.frombuffer(memoryview(arr_val), dtype=arr_info['dtype'])
                    X = X.reshape(arr_info['shape'])
                    job_info = job_id.split(b'@')
                    job_id = job_info[0]
                    partial_id = job_info[1] if len(job_info) == 2 else 0
                    pending_result[job_id].append((X, partial_id))
                    pending_checksum[job_id] += X.shape[0]
                    self.logger.info('received %d of job %s (%d/%d)' % (X.shape[0], job_id,
                                                                        pending_checksum[job_id],
                                                                        job_checksum[job_id]))

                    # check if there are finished jobs, send it back to workers
                    finished = [(k, v) for k, v in pending_result.items() if pending_checksum[k] == job_checksum[k]]
                    for job_info, tmp in finished:
                        self.logger.info(
                            'job %s %d samples are done! sending back to client' % (
                                job_info, job_checksum[job_info]))
                        # re-sort to the original order
                        tmp = [x[0] for x in sorted(tmp, key=lambda x: x[1])]
                        client_addr = job_info.split(b'#')[0]
                        send_ndarray(sender, client_addr, np.concatenate(tmp, axis=0))
                        pending_result.pop(job_info)
                        pending_checksum.pop(job_info)
                        job_checksum.pop(job_info)

                if socks.get(frontend) == zmq.POLLIN:
                    job_info, num_seqs = frontend.recv_multipart()
                    job_checksum[job_info] = int(num_seqs)
                    self.logger.info('new job %s size: %d is registered!' % (job_info, int(num_seqs)))
        except zmq.error.ContextTerminated:
            self.logger.error('context is closed!')


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
        self.logger.info('terminated!')

    def input_fn_builder(self, worker):
        def gen():
            while not self.exit_flag.is_set():
                client_id, msg = worker.recv_multipart()
                msg = jsonapi.loads(msg)
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
    return src.send_multipart([dest, jsonapi.dumps(md), X], flags, copy=copy, track=track)
