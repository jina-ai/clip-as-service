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
