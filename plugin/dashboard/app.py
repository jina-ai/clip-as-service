import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from bert_serving.client import BertClient

bs_monitor = None


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ip', type=str, default='localhost',
                        help='host address of the bert server')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    parser.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for sending result to client')
    return parser


class BertMonitor(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps(bs_monitor.server_status, ensure_ascii=False).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=BertMonitor, port=8531):
    httpd = server_class(('', port), handler_class)
    httpd.serve_forever()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    bs_monitor = BertClient(ip=args.ip,
                            port=args.port,
                            port_out=args.port_out,
                            check_version=True,
                            show_server_config=True)
    run()
