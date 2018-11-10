import sys
import time

from utils.client import EncoderClient

if __name__ == '__main__':
    ec = EncoderClient(port=sys.argv[1])
    # encode a list of strings
    with open('sample_text.txt', encoding='utf8') as fp:
        data = fp.readlines()

    for j in range(1, 200, 10):
        start_t = time.time()
        tmp = data * j
        ec.encode(tmp)
        time_t = time.time() - start_t
        print('encoding %d strs in %.2fs, speed: %d/s' %
              (len(tmp), time_t, int(len(tmp) / time_t)))
    # bad example: encode a string
    # print(ec.encode('abc'))
