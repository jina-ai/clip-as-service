from multiprocessing import Process, Event

from termcolor import colored

from .helper import set_logger


class BertHTTPProxy(Process):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.is_ready = Event()

    def create_flask_app(self):
        try:
            from flask import Flask, request
            from flask_compress import Compress
            from flask_cors import CORS
            from flask_json import FlaskJSON, as_json, JsonError
            from bert_serving.client import ConcurrentBertClient
        except ImportError:
            raise ImportError('BertClient or Flask or its dependencies are not fully installed, '
                              'they are required for serving HTTP requests.'
                              'Please use "pip install -U bert-serving-server[http]" to install it.')

        # support up to 10 concurrent HTTP requests
        bc = ConcurrentBertClient(max_concurrency=self.args.http_max_connect,
                                  port=self.args.port, port_out=self.args.port_out,
                                  output_fmt='list', ignore_all_checks=True)
        app = Flask(__name__)
        logger = set_logger(colored('PROXY', 'red'))

        @app.route('/status/server', methods=['GET'])
        @as_json
        def get_server_status():
            return bc.server_status

        @app.route('/status/client', methods=['GET'])
        @as_json
        def get_client_status():
            return bc.status

        @app.route('/encode', methods=['POST'])
        @as_json
        def encode_query():
            data = request.form if request.form else request.json
            try:
                logger.info('new request from %s' % request.remote_addr)
                return {'id': data['id'],
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
        self.is_ready.set()
        app.run(port=self.args.http_port, threaded=True, host='0.0.0.0')
