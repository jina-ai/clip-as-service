#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import multiprocessing
import os
import sys
import threading
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from termcolor import colored
from zmq.utils import jsonapi

from .bert import modeling, tokenization
from .bert.extract_features import model_fn_builder, convert_lst_to_features
from .helper import set_logger

_tf_ver = tf.__version__.split('.')
assert int(_tf_ver[0]) >= 1 and int(_tf_ver[1]) >= 10, 'Tensorflow >=1.10 is required!'

__version__ = '1.4.5'


def _auto_bind(socket):
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://*')
    else:
        socket.bind('ipc://*')
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')


class ServerCommand:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'


class BertServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'))

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
            'port_out': args.port_out,
            'pooling_layer': args.pooling_layer,
            'pooling_strategy': args.pooling_strategy.value,
            'tensorflow_version': tf.__version__,
            'python_version': sys.version,
            'server_start_time': str(datetime.now())
        }
        self.processes = []
        self.context = zmq.Context()

        # frontend facing client
        self.frontend = self.context.socket(zmq.PULL)
        self.frontend.bind('tcp://*:%d' % self.port)

        # pair connection between frontend and sink
        self.sink = self.context.socket(zmq.PAIR)
        self.addr_front2sink = _auto_bind(self.sink)

        # backend facing workers
        self.backend = self.context.socket(zmq.PUSH)
        self.addr_backend = _auto_bind(self.backend)

        # start the sink thread
        proc_sink = BertSink(self.args, self.addr_front2sink)
        proc_sink.start()
        self.processes.append(proc_sink)
        self.addr_sink = self.sink.recv().decode('ascii')

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
        run_on_gpu = True
        num_req = 0
        try:
            import GPUtil
            available_gpus = GPUtil.getAvailable(limit=self.num_worker)
            if len(available_gpus) < self.num_worker:
                self.logger.warn('only %d GPU(s) is available, but ask for %d' % (len(available_gpus), self.num_worker))
        except FileNotFoundError:
            self.logger.warn('nvidia-smi is missing, often means no gpu found on this machine. '
                             'will fall back to cpu!')
            run_on_gpu = False

        # start the backend processes
        for i in available_gpus:
            process = BertWorker(i, self.args, self.addr_backend, self.addr_sink)
            self.processes.append(process)
            process.start()

        try:
            while True:
                client, msg, req_id = self.frontend.recv_multipart()
                if msg == ServerCommand.show_config:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    self.sink.send_multipart([client, msg,
                                              jsonapi.dumps({**{'client': client.decode('ascii'),
                                                                'num_subprocess': len(self.processes),
                                                                'ventilator -> worker': self.addr_backend,
                                                                'worker -> sink': self.addr_sink,
                                                                'ventilator <-> sink': self.addr_front2sink,
                                                                'server_current_time': str(datetime.now()),
                                                                'run_on_gpu': run_on_gpu,
                                                                'num_request': num_req,
                                                                'server_version': __version__},
                                                             **self.args_dict}), req_id])
                    continue

                self.logger.info('new encode request\treq id: %d\tclient: %s' % (int(req_id), client))
                num_req += 1
                seqs = jsonapi.loads(msg)
                num_seqs = len(seqs)
                # register a new job at sink
                self.sink.send_multipart([client, ServerCommand.new_job, b'%d' % num_seqs, req_id])

                job_id = client + b'#' + req_id
                if num_seqs > self.max_batch_size:
                    # partition the large batch into small batches
                    s_idx = 0
                    while s_idx < num_seqs:
                        tmp = seqs[s_idx: (s_idx + self.max_batch_size)]
                        if tmp:
                            partial_job_id = job_id + b'@%d' % s_idx
                            self.backend.send_multipart([partial_job_id, jsonapi.dumps(tmp)])
                        s_idx += len(tmp)
                else:
                    self.backend.send_multipart([job_id, msg])
        except zmq.error.ContextTerminated:
            self.logger.error('context is closed!')


class BertSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'))
        self.front_sink_addr = front_sink_addr

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        context = zmq.Context()
        # receive from workers
        receiver = context.socket(zmq.PULL)
        receiver_addr = _auto_bind(receiver)

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
        frontend.send(receiver_addr.encode('ascii'))

        try:
            while not self.exit_flag.is_set():
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
                    self.logger.info('collect job %s (%d/%d)' % (job_id,
                                                                 pending_checksum[job_id],
                                                                 job_checksum[job_id]))

                    # check if there are finished jobs, send it back to workers
                    finished = [(k, v) for k, v in pending_result.items() if pending_checksum[k] == job_checksum[k]]
                    for job_info, tmp in finished:
                        self.logger.info(
                            'send back\tsize: %d\tjob id:%s\t' % (
                                job_checksum[job_info], job_info))
                        # re-sort to the original order
                        tmp = [x[0] for x in sorted(tmp, key=lambda x: x[1])]
                        client_addr, req_id = job_info.split(b'#')
                        send_ndarray(sender, client_addr, np.concatenate(tmp, axis=0), req_id)
                        pending_result.pop(job_info)
                        pending_checksum.pop(job_info)
                        job_checksum.pop(job_info)

                if socks.get(frontend) == zmq.POLLIN:
                    client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                    if msg_type == ServerCommand.new_job:
                        job_info = client_addr + b'#' + req_id
                        job_checksum[job_info] = int(msg_info)
                        self.logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                    elif msg_type == ServerCommand.show_config:
                        sender.send_multipart([client_addr, msg_info, req_id])
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
        self.estimator = Estimator(self.model_fn, config=RunConfig(session_config=config))
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'))
        self.worker_address = worker_address
        self.sink_address = sink_address
        self.prefetch_factor = 10

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

        for r in self.estimator.predict(input_fn, yield_single_examples=False):
            send_ndarray(sink, r['client_id'], r['encodes'])
            self.logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))

        receiver.close()
        sink.close()
        context.term()
        self.logger.info('terminated!')

    def input_fn_builder(self, worker):
        def gen():
            self.logger.info('ready and listening!')
            while not self.exit_flag.is_set():
                client_id, msg = worker.recv_multipart()
                msg = jsonapi.loads(msg)
                self.logger.info('new job\tsize: %d\tclient: %s' % (len(msg), client_id))
                tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, self.tokenizer))
                yield {
                    'client_id': client_id,
                    'input_ids': [f.input_ids for f in tmp_f],
                    'input_mask': [f.input_mask for f in tmp_f],
                    'input_type_ids': [f.input_type_ids for f in tmp_f]
                }

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
                    'input_type_ids': (None, self.max_seq_len)}).prefetch(self.prefetch_factor))

        return input_fn


def send_ndarray(src, dest, X, req_id=b'', flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype=str(X.dtype), shape=X.shape)
    return src.send_multipart([dest, jsonapi.dumps(md), X, req_id], flags, copy=copy, track=track)
