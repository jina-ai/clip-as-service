import logging
import pickle
import threading
import time

import zmq

from gpu_env import APP_NAME


class ServerTask(threading.Thread):
    """ServerTask"""

    def __init__(self, args, port=5555):
        threading.Thread.__init__(self)
        self.args = args
        self.port = port

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:%d' % self.port)

        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend')

        workers = []
        for id in range(self.args.num_server):
            worker = ServerWorker(context, self.args, id)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()


class ServerWorker(threading.Thread):
    """ServerWorker"""

    def __init__(self, context, args, id):
        threading.Thread.__init__(self)
        self.context = context
        self.args = args
        self.id = id

    def is_valid_input(self, texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)

    def run(self):
        logger = logging.getLogger(APP_NAME)
        worker = self.context.socket(zmq.DEALER)
        worker.connect('inproc://backend')
        model = build_model(self.args)
        model.restore()
        while True:
            ident, msg = worker.recv_multipart()
            start_t = time.time()
            msg = pickle.loads(msg)
            if self.is_valid_input(msg):
                worker.send_multipart([ident, pickle.dumps(model.predict(msg))])
                logger.info('worker %d: encoding %d strings in %.4fs speed: %d/s' % (self.id,
                                                                                     len(msg), time.time() - start_t,
                                                                                     int(len(msg) / (
                                                                                             time.time() - start_t))))
            else:
                logger.warning('worker %d: received unsupported type! sending back None' % self.id)
                worker.send_multipart([ident, pickle.dumps(None)])
        worker.close()
