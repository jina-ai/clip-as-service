import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from bert_serving.client import BertClient

ip = 'localhost'
port = 4000
port_out = 4001


class BertMonitor(BaseHTTPRequestHandler, BertClient):

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)
        self.bc = BertClient(port=4000, port_out=4001, check_version=True, show_server_config=True)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps(self.bc.server_status))


def run(server_class=HTTPServer, handler_class=BertMonitor, port=8531):
    httpd = server_class(('', port), handler_class)
    httpd.serve_forever()


if __name__ == "__main__":
    run()
