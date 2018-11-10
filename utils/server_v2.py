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

tf.logging.set_verbosity(tf.logging.INFO)


def is_valid_input(texts):
    return isinstance(texts, list) and all(isinstance(s, str) for s in texts)


class ServerTask(threading.Thread):
    """ServerTask"""

    def __init__(self, args):
        super().__init__()
        self.model_dir = args.model_dir
        self.max_seq_len = args.max_len
        self.num_server = args.num_server
        self.port = args.port

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:%d' % self.port)

        backend = context.socket(zmq.DEALER)
        backend.bind('ipc:///tmp/backend0')

        workers = []
        for id in range(self.num_server):
            worker = ServerWorker(context, id, self.model_dir, self.max_seq_len, id + 1)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()


class ServerWorker(Process):
    """ServerWorker"""

    def __init__(self, context, id, model_dir, max_seq_len, gpu_id):
        super().__init__()
        self.context = context
        self.model_dir = model_dir
        self.config_fp = os.path.join(self.model_dir, 'bert_config.json')
        self.checkpoint_fp = os.path.join(self.model_dir, 'bert_model.ckpt')
        self.vocab_fp = os.path.join(model_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_fp)
        self.max_seq_len = max_seq_len
        self.id = id
        self.model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(self.config_fp),
            init_checkpoint=self.checkpoint_fp)
        # session_config = tf.ConfigProto()
        # session_config.gpu_options.visible_device_list = '%d' % gpu_id
        # run_config = tf.estimator.RunConfig(session_config=session_config)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.estimator = Estimator(self.model_fn)
        self.result = []

    def run(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect('ipc:///tmp/backend0')
        input_fn = self.input_fn_builder(worker)
        logger.info('worker %d is ready and listening' % self.id)
        for r in self.estimator.predict(input_fn):
            self.result.append([round(float(x), 6) for x in r.flat])
        worker.close()

    def input_fn_builder(self, worker):
        def gen():
            while True:
                if self.result:
                    num_result = len(self.result)
                    worker.send_multipart([ident, pickle.dumps(self.result)])
                    self.result = []
                    time_used = time.clock() - start
                    logger.info('encoded %d strs from %s in %.2fs @ %d/s' %
                                (num_result, ident, time_used,
                                 int(num_result / time_used)))
                ident, msg = worker.recv_multipart()
                start = time.clock()
                msg = pickle.loads(msg)
                if is_valid_input(msg):
                    tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, self.tokenizer))
                    yield {
                        'input_ids': [f.input_ids for f in tmp_f],
                        'input_mask': [f.input_mask for f in tmp_f],
                        'input_type_ids': [f.input_type_ids for f in tmp_f]
                    }
                else:
                    logger.warning('worker %d: received unsupported type! sending back None' % self.id)
                    worker.send_multipart([ident, pickle.dumps(None)])

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={k: tf.int32 for k in ['input_ids', 'input_mask', 'input_type_ids']},
                output_shapes={'input_ids': (None, self.max_seq_len),
                               'input_mask': (None, self.max_seq_len),
                               'input_type_ids': (None, self.max_seq_len)}))

        return input_fn
