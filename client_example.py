import time


class EncoderClient:
    def __init__(self, ip='localhost', port=5555):
        import zmq
        from datetime import datetime
        context = zmq.Context()
        self.socket = context.socket(zmq.DEALER)
        self.socket.identity = ('client-%d' % datetime.now().timestamp()).encode('ascii')
        self.socket.connect('tcp://%s:%d' % (ip, port))

    def encode(self, texts):
        if self.is_valid_input(texts):
            self.socket.send_pyobj(texts)
            return self.socket.recv_pyobj()
        else:
            raise AttributeError('"texts" must be List[str]!')

    @staticmethod
    def is_valid_input(texts):
        return isinstance(texts, list) and all(isinstance(s, str) for s in texts)


if __name__ == '__main__':
    ec = EncoderClient('100.102.33.53')
    # encode a list of strings
    with open('sample_text.txt', encoding='utf8') as fp:
        data = fp.readlines()

    for j in range(1, 100, 10):
        start_t = time.time()
        ec.encode(data * j)
        time_t = time.time() - start_t
        print('encoding %d strs in %.3fs, speed: %d/s' %
              (len(data * j), time_t, int(len(data * j) / time_t)))
    # bad example: encode a string
    # print(ec.encode('abc'))
