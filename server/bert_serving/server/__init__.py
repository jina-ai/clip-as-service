#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import multiprocessing
import os
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from multiprocessing import Process
from multiprocessing.pool import Pool

import numpy as np
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from .helper import *
from .http import BertHTTPProxy
from .zmq_decor import multi_socket

__all__ = ['__version__', 'BertServer']
__version__ = '1.9.3'

_tf_ver_ = check_tf_version()


class ServerCmd:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    show_status = b'SHOW_STATUS'
    new_job = b'REGISTER'
    data_token = b'TOKENS'
    data_embed = b'EMBEDDINGS'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


class BertServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        self.model_dir = args.model_dir
        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port
        self.args = args
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'tensorflow_version': _tf_ver_,
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.processes = []
        self.logger.info('freeze, optimize and export graph, could take a while...')
        with Pool(processes=1) as pool:
            # optimize the graph, must be done in another process
            from .graph import optimize_graph
            self.graph_path, self.bert_config = pool.apply(optimize_graph, (self.args,))
        # from .graph import optimize_graph
        # self.graph_path = optimize_graph(self.args, self.logger)
        if self.graph_path:
            self.logger.info('optimized graph is stored at: %s' % self.graph_path)
        else:
            raise FileNotFoundError('graph optimization fails and returns empty result')
        self.is_ready = threading.Event()

    def __enter__(self):
        self.start()
        self.is_ready.wait()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        self.is_ready.clear()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'0', ServerCmd.terminate, b'0', b'0'])

    @staticmethod
    def shutdown(args):
        with zmq.Context() as ctx:
            ctx.setsockopt(zmq.LINGER, args.timeout)
            with ctx.socket(zmq.PUSH) as frontend:
                try:
                    frontend.connect('tcp://%s:%d' % (args.ip, args.port))
                    frontend.send_multipart([b'0', ServerCmd.terminate, b'0', b'0'])
                    print('shutdown signal sent to %d' % args.port)
                except zmq.error.Again:
                    raise TimeoutError(
                        'no response from the server (with "timeout"=%d ms), please check the following:'
                        'is the server still online? is the network broken? are "port" correct? ' % args.timeout)

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            # backend_socks[0] is always at the highest priority
            _sock = backend_socks[0] if _msg_len <= self.args.priority_batch_size else rand_backend_socket
            _sock.send_multipart([_job_id, _json_msg])

        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets' % len(addr_backend_list))

        # start the sink process
        self.logger.info('start the sink')
        proc_sink = BertSink(self.args, addr_front2sink, self.bert_config)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            process = BertWorker(idx, self.args, addr_backend_list, addr_sink, device_id,
                                 self.graph_path, self.bert_config)
            self.processes.append(process)
            process.start()

        # start the http-service process
        if self.args.http_port:
            self.logger.info('start http proxy')
            proc_proxy = BertHTTPProxy(self.args)
            self.processes.append(proc_proxy)
            proc_proxy.start()

        rand_backend_socket = None
        server_status = ServerStatistic()

        for p in self.processes:
            p.is_ready.wait()

        self.is_ready.set()
        self.logger.info('all set, ready to serve request!')

        while True:
            try:
                request = frontend.recv_multipart()
                client, msg, req_id, msg_len = request
                assert req_id.isdigit()
                assert msg_len.isdigit()
            except (ValueError, AssertionError):
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg == ServerCmd.terminate:
                    break
                elif msg == ServerCmd.show_config or msg == ServerCmd.show_status:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    status_runtime = {'client': client.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'ventilator -> worker': addr_backend_list,
                                      'worker -> sink': addr_sink,
                                      'ventilator <-> sink': addr_front2sink,
                                      'server_current_time': str(datetime.now()),
                                      'device_map': device_map,
                                      'num_concurrent_socket': self.num_concurrent_socket}
                    if msg == ServerCmd.show_status:
                        status_runtime['statistic'] = server_status.value
                    sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime,
                                                                     **self.status_args,
                                                                     **self.status_static}), req_id])
                else:
                    self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    # register a new job at sink
                    sink.send_multipart([client, ServerCmd.new_job, msg_len, req_id])

                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backennd shouldn't be selected either as it may be queued up already
                    rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

                    # push a new job, note super large job will be pushed to one socket only,
                    # leaving other sockets free
                    job_id = client + b'#' + req_id
                    if int(msg_len) > self.max_batch_size:
                        seqs = jsonapi.loads(msg)
                        job_gen = ((job_id + b'@%d' % i, seqs[i:(i + self.max_batch_size)]) for i in
                                   range(0, int(msg_len), self.max_batch_size))
                        for partial_job_id, job in job_gen:
                            push_new_job(partial_job_id, jsonapi.dumps(job), len(job))
                    else:
                        push_new_job(job_id, msg, int(msg_len))

        for p in self.processes:
            p.close()
        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker),
                                                maxMemory=0.9, maxLoad=0.9)
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map


class BertSink(Process):
    def __init__(self, args, front_sink_addr, bert_config):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'), args.verbose)
        self.front_sink_addr = front_sink_addr
        self.verbose = args.verbose
        self.show_tokens_to_client = args.show_tokens_to_client
        self.max_seq_len = args.max_seq_len
        self.max_position_embeddings = bert_config.max_position_embeddings
        self.fixed_embed_length = args.fixed_embed_length
        self.is_ready = multiprocessing.Event()

    def close(self):
        self.logger.info('shutting down...')
        self.is_ready.clear()
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_jobs = defaultdict(lambda: SinkJob(self.max_seq_len, self.max_position_embeddings,
                                                   self.show_tokens_to_client,
                                                   self.fixed_embed_length))  # type: Dict[str, SinkJob]

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))

        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compability
        logger = set_logger(colored('SINK', 'green'), self.verbose)
        logger.info('ready')
        self.is_ready.set()

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                # parsing job_id and partial_id
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = int(job_info[1]) if len(job_info) == 2 else 0

                if msg[3] == ServerCmd.data_embed:
                    # parsing the ndarray
                    arr_info, arr_val = jsonapi.loads(msg[1]), msg[2]
                    x = np.frombuffer(memoryview(arr_val), dtype=arr_info['dtype']).reshape(arr_info['shape'])
                    pending_jobs[job_id].add_embed(x, partial_id)
                elif msg[3] == ServerCmd.data_token:
                    x = jsonapi.loads(msg[1])
                    pending_jobs[job_id].add_token(x, partial_id)
                else:
                    logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(msg))
                    logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(msg)), exc_info=True)

                logger.info('collect %s %s (E:%d/T:%d/A:%d)' % (msg[3], job_id,
                                                                pending_jobs[job_id].progress_embeds,
                                                                pending_jobs[job_id].progress_tokens,
                                                                pending_jobs[job_id].checksum))

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCmd.new_job:
                    job_info = client_addr + b'#' + req_id
                    # register a new job
                    pending_jobs[job_info].checksum = int(msg_info)
                    logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                    if len(pending_jobs[job_info]._pending_embeds)>0 \
                            and pending_jobs[job_info].final_ndarray is None:
                        pending_jobs[job_info].add_embed(None, 0)
                elif msg_type == ServerCmd.show_config or msg_type == ServerCmd.show_status:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])

            # check if there are finished jobs, then send it back to workers
            finished = [(k, v) for k, v in pending_jobs.items() if v.is_done]
            for job_info, tmp in finished:
                client_addr, req_id = job_info.split(b'#')
                x, x_info = tmp.result
                sender.send_multipart([client_addr, x_info, x, req_id])
                logger.info('send back\tsize: %d\tjob id: %s' % (tmp.checksum, job_info))
                # release the job
                tmp.clear()
                pending_jobs.pop(job_info)


class SinkJob:
    def __init__(self, max_seq_len, max_position_embeddings, with_tokens, fixed_embed_length):
        self._pending_embeds = []
        self.tokens = []
        self.tokens_ids = []
        self.checksum = 0
        self.final_ndarray = None
        self.progress_tokens = 0
        self.progress_embeds = 0
        self.with_tokens = with_tokens
        self.max_seq_len_unset = max_seq_len is None
        self.max_position_embeddings = max_position_embeddings
        self.max_effective_len = 0
        self.fixed_embed_length = fixed_embed_length

    def clear(self):
        self._pending_embeds.clear()
        self.tokens_ids.clear()
        self.tokens.clear()
        del self.final_ndarray

    def _insert(self, data, pid, data_lst, idx_lst):
        lo = 0
        hi = len(idx_lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if pid < idx_lst[mid]:
                hi = mid
            else:
                lo = mid + 1
        idx_lst.insert(lo, pid)
        data_lst.insert(lo, data)

    def add_embed(self, data, pid):
        def fill_data():
            self.final_ndarray[pid: (pid + data.shape[0]), 0:data.shape[1]] = data
            self.progress_embeds += progress
            if data.shape[1] > self.max_effective_len:
                self.max_effective_len = data.shape[1]
        if data is not None: # when job finish msg come to SINK earlier than job register
            progress = data.shape[0]
        else:
            progress = 0
        if not self.checksum:
            self._pending_embeds.append((data, pid, progress))
        else:
            if self.final_ndarray is None:
                if data is not None: # when job finish msg come to SINK earlier than job register
                    d_shape = list(data.shape[1:])
                else:
                    d_shape = list(self._pending_embeds[0][0].shape[1:])
                if self.max_seq_len_unset and len(d_shape) > 1:
                    # if not set max_seq_len, then we have no choice but set result ndarray to
                    # [B, max_position_embeddings, dim] and truncate it at the end
                    d_shape[0] = self.max_position_embeddings
                if data is not None:
                    dtype = data.dtype
                else:
                    dtype = self._pending_embeds[0][0].dtype
                self.final_ndarray = np.zeros([self.checksum] + d_shape, dtype=dtype)
            if data is not None: # when job finish msg come to SINK earlier than job register
                fill_data()
            while self._pending_embeds:
                data, pid, progress = self._pending_embeds.pop()
                fill_data()

    def add_token(self, data, pid):
        progress = len(data)
        self._insert(data, pid, self.tokens, self.tokens_ids)
        self.progress_tokens += progress

    @property
    def is_done(self):
        if self.with_tokens:
            return self.checksum > 0 and self.checksum == self.progress_tokens and self.checksum == self.progress_embeds
        else:
            return self.checksum > 0 and self.checksum == self.progress_embeds

    @property
    def result(self):
        if self.max_seq_len_unset and not self.fixed_embed_length:
            x = np.ascontiguousarray(self.final_ndarray[:, 0:self.max_effective_len])
        else:
            x = self.final_ndarray
        x_info = {'dtype': str(x.dtype),
                  'shape': x.shape,
                  'tokens': list(chain.from_iterable(self.tokens)) if self.with_tokens else ''}

        x_info = jsonapi.dumps(x_info)
        return x, x_info


class BertWorker(Process):
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, graph_config):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
        self.max_seq_len = args.max_seq_len
        self.do_lower_case = args.do_lower_case
        self.mask_cls_sep = args.mask_cls_sep
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.model_dir = args.model_dir
        self.verbose = args.verbose
        self.graph_path = graph_path
        self.bert_config = graph_config
        self.use_fp16 = args.fp16
        self.show_tokens_to_client = args.show_tokens_to_client
        self.is_ready = multiprocessing.Event()

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.is_ready.clear()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def get_estimator(self, tf):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'encodes': output[0]
            })

        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink_embed, sink_token, *receivers):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)

        logger.info('use device %s, load graph from %s' %
                    ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))

        tf = import_tf(self.device_id, self.verbose, use_fp16=self.use_fp16)
        estimator = self.get_estimator(tf)

        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink_embed.connect(self.sink_address)
        sink_token.connect(self.sink_address)
        for r in estimator.predict(self.input_fn_builder(receivers, tf, sink_token), yield_single_examples=False):
            send_ndarray(sink_embed, r['client_id'], r['encodes'], ServerCmd.data_embed)
            logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))

    def input_fn_builder(self, socks, tf, sink):
        from .bert.extract_features import convert_lst_to_features
        from .bert.tokenization import FullTokenizer

        def gen():
            # Windows does not support logger in MP environment, thus get a new logger
            # inside the process for better compatibility
            logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
            tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'), do_lower_case=self.do_lower_case)

            poller = zmq.Poller()
            for sock in socks:
                poller.register(sock, zmq.POLLIN)

            logger.info('ready and listening!')
            self.is_ready.set()

            while not self.exit_flag.is_set():
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(socks):
                    if sock in events:
                        client_id, raw_msg = sock.recv_multipart()
                        msg = jsonapi.loads(raw_msg)
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg), client_id))
                        # check if msg is a list of list, if yes consider the input is already tokenized
                        is_tokenized = all(isinstance(el, list) for el in msg)
                        tmp_f = list(convert_lst_to_features(msg, self.max_seq_len,
                                                             self.bert_config.max_position_embeddings,
                                                             tokenizer, logger,
                                                             is_tokenized, self.mask_cls_sep))
                        if self.show_tokens_to_client:
                            sink.send_multipart([client_id, jsonapi.dumps([f.tokens for f in tmp_f]),
                                                 b'', ServerCmd.data_token])
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
                    'input_ids': (None, None),
                    'input_mask': (None, None),
                    'input_type_ids': (None, None)}).prefetch(self.prefetch_size))

        return input_fn


class ServerStatistic:
    def __init__(self):
        self._hist_client = CappedHistogram(500)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = CappedHistogram(500)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        client, msg, req_id, msg_len = request
        self._hist_client[client] += 1
        if ServerCmd.is_valid(msg):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp

    @property
    def value(self):
        def get_min_max_avg(name, stat, avg=None):
            if len(stat) > 0:
                avg = sum(stat) / len(stat) if avg is None else avg
                min_, max_ = min(stat), max(stat)
                return {
                    'avg_%s' % name: avg,
                    'min_%s' % name: min_,
                    'max_%s' % name: max_,
                    'num_min_%s' % name: sum(v == min_ for v in stat),
                    'num_max_%s' % name: sum(v == max_ for v in stat),
                }
            else:
                return {}

        def get_num_active_client(interval=180):
            # we count a client active when its last request is within 3 min.
            now = time.perf_counter()
            return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

        avg_msg_len = None
        if len(self._hist_msg_len) > 0:
            avg_msg_len = sum(k*v for k,v in self._hist_msg_len.items()) / sum(self._hist_msg_len.values())

        parts = [{
            'num_data_request': self._num_data_req,
            'num_total_seq': self._num_total_seq,
            'num_sys_request': self._num_sys_req,
            'num_total_request': self._num_data_req + self._num_sys_req,
            'num_total_client': self._hist_client.total_size(),
            'num_active_client': get_num_active_client()},
            self._hist_client.get_stat_map('request_per_client'),
            get_min_max_avg('size_per_request', self._hist_msg_len.keys(), avg=avg_msg_len),
            get_min_max_avg('last_two_interval', self._last_two_req_interval),
            get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
        ]

        return {k: v for d in parts for k, v in d.items()}
