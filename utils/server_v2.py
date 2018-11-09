import os
import pickle
import threading

import tensorflow as tf
import zmq
from tensorflow.python.estimator.estimator import Estimator

import modeling
import tokenization
from extract_features import model_fn_builder, convert_lst_to_features
from utils.helper import set_logger

logger = set_logger()


def is_valid_input(texts):
    return isinstance(texts, list) and all(isinstance(s, str) for s in texts)


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
        self.result = []

    def run(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect('inproc://backend')
        input_fn = self.input_fn_builder(worker)
        logger.info('worker %d is ready and listening' % self.id)
        for r in self.estimator.predict(input_fn):
            logger.info('add new result')
            self.result.append([round(float(x), 8) for x in r['unique_id'].flat])
        worker.close()

    def input_fn_builder(self, worker):
        def gen():
            while True:
                if self.result:
                    worker.send_multipart([ident, pickle.dumps(self.result)])
                    self.result = []
                ident, msg = worker.recv_multipart()
                msg = pickle.loads(msg)
                logger.info('received new data!')
                if is_valid_input(msg):
                    for f in convert_lst_to_features(msg, self.max_seq_len, self.tokenizer):
                        logger.info('yield new sample')
                        yield {
                            'unique_ids': f.unique_id,
                            'input_ids': f.input_ids,
                            'input_mask': f.input_mask,
                            'input_type_ids': f.input_type_ids
                        }
                else:
                    logger.warning('worker %d: received unsupported type! sending back None' % self.id)
                    worker.send_multipart([ident, pickle.dumps(None)])

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={k: tf.int32
                              for k in ['unique_ids', 'input_ids', 'input_mask',
                                        'input_type_ids']},
                output_shapes={'unique_ids': (),
                               'input_ids': (self.max_seq_len,),
                               'input_mask': (self.max_seq_len,),
                               'input_type_ids': (self.max_seq_len,)})
                    .batch(1))

        return input_fn
