import os
import pickle
import threading
import time
from multiprocessing import Process

import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator

import modeling
import tokenization
from extract_features import model_fn_builder, convert_lst_to_features
from utils.helper import set_logger

logger = set_logger()


class ServerTask(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.model_dir = args.model_dir
        self.max_len = args.max_len
        self.num_worker = args.num_worker
        self.port = args.port
        self.args = args

    def run(self):
        context = zmq.Context.instance()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:%d' % self.port)
        backend = context.socket(zmq.ROUTER)
        backend.bind('ipc:///tmp/bert.service')

        for i in range(self.num_worker):
            process = ServerWorker(i, self.args)
            process.start()

        # Initialize main loop state
        workers = []
        poller = zmq.Poller()
        # Only poll for requests from backend until workers are available
        poller.register(backend, zmq.POLLIN)

        while True:
            logger.info('available workers: %d' % len(workers))
            sockets = dict(poller.poll())

            if backend in sockets:
                # Handle worker activity on the backend
                request = backend.recv_multipart()
                worker, empty, client = request[:3]
                if not workers:
                    # Poll for clients now that a worker is available
                    poller.register(frontend, zmq.POLLIN)
                workers.append(worker)
                if client != b'READY' and len(request) > 3:
                    # If client reply, send rest back to frontend
                    empty, reply = request[3:]
                    frontend.send_multipart([client, b'', reply])

            if frontend in sockets:
                # Get next client request, route to last-used worker
                client, empty, request = frontend.recv_multipart()
                worker = workers.pop(0)
                backend.send_multipart([worker, b'', client, b'', request])
                if not workers:
                    # Don't poll clients if no workers are available
                    poller.unregister(frontend)

        frontend.close()
        backend.close()
        context.term()


class ServerWorker(Process):
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
        # session_config = tf.ConfigProto()
        # session_config.gpu_options.visible_device_list = '%d' % gpu_id
        # run_config = tf.estimator.RunConfig(session_config=session_config)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.worker_id)
        self.estimator = Estimator(self.model_fn)
        self.result = []

    def run(self):
        socket = zmq.Context().socket(zmq.REQ)
        socket.identity = u'Worker-{}'.format(self.worker_id).encode('ascii')
        socket.connect('ipc:///tmp/bert.service')

        input_fn = self.input_fn_builder(socket)
        socket.send(b'READY')
        logger.info('worker %d is ready and listening' % self.worker_id)
        for r in self.estimator.predict(input_fn):
            self.result.append([round(float(x), 6) for x in r.flat])
        socket.close()
        logger.info('closed!')

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)

    def input_fn_builder(self, worker):
        def gen():
            while True:
                if self.result:
                    num_result = len(self.result)
                    worker.send_multipart([ident, b'', pickle.dumps(self.result)])
                    self.result = []
                    time_used = time.clock() - start
                    logger.info('encoded %d strs from %s in %.2fs @ %d/s' %
                                (num_result, ident, time_used,
                                 int(num_result / time_used)))
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
