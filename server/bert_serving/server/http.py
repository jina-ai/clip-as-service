from multiprocessing import Process

from termcolor import colored

from .helper import set_logger


class BertHTTPProxy(Process):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def create_flask_app(self):
        try:
            from flask import Flask, request
            from flask_compress import Compress
            from flask_cors import CORS
            from flask_json import FlaskJSON, as_json, JsonError
        except ImportError:
            raise ImportError('Flask or its dependencies are not fully installed, '
                              'they are required for serving HTTP requests.'
                              'Please use "pip install -U flask flask-compress flask-cors flask-json" to install it.')

        # support up to 10 concurrent HTTP requests
        bert_client = ConcurrentBertClient(self.args, num_concurrent=self.args.http_max_connect)
        app = Flask(__name__)
        logger = set_logger(colored('PROXY', 'red'))

        @app.route('/status', methods=['GET'])
        @as_json
        def get_all_categories():
            logger.info('return server status')
            with bert_client as bc:
                return bc.server_status

        @app.route('/encode', methods=['POST'])
        @as_json
        def _update_product():
            data = request.form if request.form else request.json
            try:
                logger.info('new request from %s' % request.remote_addr)
                with bert_client as bc:
                    return {
                        'id': data['id'],
                        'result': bc.encode(data['texts'], is_tokenized=bool(
                            data['is_tokenized']) if 'is_tokenized' in data else False)}

            except Exception as e:
                logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        CORS(app, origins=self.args.cors)
        FlaskJSON(app)
        Compress().init_app(app)
        return app

    def run(self):
        app = self.create_flask_app()
        app.run(port=self.args.http_port, threaded=True, host='0.0.0.0')


class ConcurrentBertClient:
    def __init__(self, args, num_concurrent=10):
        try:
            from bert_serving.client import BertClient
        except ImportError:
            raise ImportError('BertClient module is not available, it is required for serving HTTP requests.'
                              'Please use "pip install -U bert-serving-client" to install it.'
                              'If you do not want to use it as an HTTP server, '
                              'then remove "-http_port" from the command line.')

        self.bc_list = [BertClient(port=args.port, port_out=args.port_out, output_fmt='list')
                        for _ in range(num_concurrent)]
        self.hanging_bc = None

    def __enter__(self):
        self.hanging_bc = self.bc_list.pop()
        return self.hanging_bc

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hanging_bc:
            self.bc_list.append(self.hanging_bc)
        self.hanging_bc = None
