# using BertClient in multi-cast way

import sys
import threading

from service.client import BertClient


def client_clone(id, idx):
    bc = BertClient(ip='localhost', port=int(sys.argv[1]), port_out=int(sys.argv[2]),
                    identity=id)
    for j in bc.listen():
        print('clone-client-%d: received %d x %d' % (idx, j.shape[0], j.shape[0]))


if __name__ == '__main__':
    bc = BertClient(ip='localhost', port=int(sys.argv[1]), port_out=int(sys.argv[2]))
    t1 = threading.Thread(target=client_clone, args=(bc.identity, 1))
    t2 = threading.Thread(target=client_clone, args=(bc.identity, 2))
    t1.start()
    t2.start()

    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]

    for _ in range(3):
        bc.encode(data)
