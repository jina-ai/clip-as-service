from multiprocessing import Process

from termcolor import colored

from .helper import set_logger


class BertHTTPProxy(Process):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def create_flask_app(self):
        try:
            from bert_serving.client import BertClient
        except ImportError:
            raise ImportError('BertClient module is not available, it is required for serving HTTP requests.'
                              'Please use "pip install -U bert-serving-client" to install it.'
                              'If you do not want to use it as an HTTP server, '
                              'then remove "-http_port" from the command line.')
        try:
            from flask import Flask, request
            from flask_compress import Compress
            from flask_cors import CORS
            from flask_json import FlaskJSON, as_json, JsonError
        except ImportError:
            raise ImportError('Flask or its dependencies are not fully installed, '
                              'they are required for serving HTTP requests.'
                              'Please use "pip install -U flask flask-compress flask-cors flask-json" to install it.')

        bc = BertClient(port=self.args.port, port_out=self.args.port_out, output_fmt='list')
        app = Flask(__name__)
        logger = set_logger(colored('PROXY', 'red'))

        @app.route('/status', methods=['GET'])
        @as_json
        def get_all_categories():
            logger.info('return server status')
            return bc.server_status

        @app.route('/encode', methods=['POST'])
        @as_json
        def _update_product():
            data = request.form if request.form else request.json
            try:
                logger.info('new request from %s' % request.remote_addr)
                return {
                    'id': data['id'],
                    'result': bc.encode(data['texts'],
                                        is_tokenized=bool(data['is_tokenized']) if 'is_tokenized' in data else False)}
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
