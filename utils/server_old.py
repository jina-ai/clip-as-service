import os
import pickle
import threading
import time

import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator

import modeling
import tokenization
from extract_features import model_fn_builder, convert_lst_to_features
from utils.helper import set_logger, JobContext

logger = set_logger()


def input_fn_builder(msg, seq_length, batch_size, tokenizer):
    def gen():
        for f in convert_lst_to_features(msg, seq_length, tokenizer):
            yield {
                'unique_ids': f.unique_id,
                'input_ids': f.input_ids,
                'input_mask': f.input_mask,
                'input_type_ids': f.input_type_ids
            }

    def input_fn():
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={k: tf.int32
                          for k in ['unique_ids', 'input_ids', 'input_mask',
                                    'input_type_ids']},
            output_shapes={'unique_ids': (),
                           'input_ids': (seq_length,),
                           'input_mask': (seq_length,),
                           'input_type_ids': (seq_length,)})
                .batch(batch_size))

    return input_fn


class ServerTask(threading.Thread):
    """ServerTask"""

    def __init__(self, model_dir, num_server=2,
                 max_seq_len=200, batch_size=128, port=5555):
        threading.Thread.__init__(self)
        self.model_dir = model_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_server = num_server
        self.port = port

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:%d' % self.port)

        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend')

        workers = []
        for id in range(self.num_server):
            worker = ServerWorker(context, id, self.model_dir, self.max_seq_len, self.batch_size)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()


class ServerWorker(threading.Thread):
    """ServerWorker"""

    def __init__(self, context, id, model_dir, max_seq_len, batch_size):
        threading.Thread.__init__(self)
        self.context = context
        self.model_dir = model_dir
        self.config_fp = os.path.join(self.model_dir, 'bert_config.json')
        self.checkpoint_fp = os.path.join(self.model_dir, 'bert_model.ckpt')
        self.vocab_fp = os.path.join(model_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_fp)
        self.max_seq_len = max_seq_len
        self.id = id
        self.batch_size = batch_size
        self.model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(self.config_fp),
            init_checkpoint=self.checkpoint_fp)
        self.estimator = Estimator(self.model_fn)

    def is_valid_input(self, texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)

    def run(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect('inproc://backend')
        logger.info('worker %d is ready and listening' % self.id)
        while True:
            ident, msg = worker.recv_multipart()
            start_t = time.time()
            with JobContext('pickle.loads'):
                msg = pickle.loads(msg)

            if self.is_valid_input(msg):
                input_fn = input_fn_builder(msg, self.max_seq_len, self.batch_size, self.tokenizer)
                result = []
                with JobContext('predict'):
                    for r in self.estimator.predict(input_fn):
                        result.append([round(float(x), 8) for x in r['unique_id'].flat])

                with JobContext('send back'):
                    worker.send_multipart([ident, pickle.dumps(result)])
                logger.info('worker %d: '
                            'encoding %d strings '
                            'in %.4fs speed: %d/s' % (self.id,
                                                      len(msg), time.time() - start_t,
                                                      int(len(msg) / (time.time() - start_t))))
            else:
                logger.warning('worker %d: received unsupported type! sending back None' % self.id)
                worker.send_multipart([ident, pickle.dumps(None)])
        worker.close()
