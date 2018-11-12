import time

from service.client import BertClient

if __name__ == '__main__':
    ec = BertClient(ip='localhost', port=5555)
    # encode a list of strings
    with open('README.md', encoding='utf8') as fp:
        data = [v for v in fp if v.strip()]

    for j in range(1, 200, 10):
        start_t = time.time()
        tmp = data * j
        ec.encode(tmp)
        time_t = time.time() - start_t
        print('encoding %d strs in %.2fs, speed: %d/s' %
              (len(tmp), time_t, int(len(tmp) / time_t)))
    # bad example: encode a string
    # print(ec.encode('abc'))
