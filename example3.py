import sys

from service.client import BertClient

if __name__ == '__main__':
    bc = BertClient(ip='localhost', port=int(sys.argv[1]), port_out=int(sys.argv[2]))
    # encode a list of strings
    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]


    # a endless data stream
    def text_gen():
        while True:
            yield data


    for j in bc.encode_async(text_gen()):
        print(j.shape[0])
