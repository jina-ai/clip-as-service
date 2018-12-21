import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import BertClient


class BertMonitor(BaseHTTPRequestHandler, BertClient):
    def __init__(self, request, client_address, server, ip='localhost', port=5555, port_out=5556):
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)
        BertClient.__init__(self, ip, port, port_out, check_version=True, show_server_config=True)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps(self.server_status))


def run(server_class=HTTPServer, handler_class=BertMonitor, port=8531):
    httpd = server_class(('', port), handler_class)
    httpd.serve_forever()


if __name__ == "__main__":
    run()
