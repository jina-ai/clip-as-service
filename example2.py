# using BertClient in async way

import sys

from service.client import BertClient


# an endless data stream, generating data in an extremely fast speed
def text_gen():
    while True:
        yield data


if __name__ == '__main__':
    bc = BertClient(ip='localhost', port=int(sys.argv[1]), port_out=int(sys.argv[2]))

    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]

    # get encoded vectors
    for j in bc.encode_async(text_gen(), max_num_batch=10):
        print('received %s' % j.shape)
