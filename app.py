import argparse

try:
    import gpu_env
except:
    print('no GPUutils!')
from utils.server import ServerTask


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='/data/cips/save/chinese-bert/chinese_L-12_H-768_A-12/',
                        help='pretrained BERT model')
    parser.add_argument('-max_len', type=int, default=25,
                        help='maximum length of a sequence')
    parser.add_argument('-num_server', type=int, default=2,
                        help='number of server instances')
    parser.add_argument('-port', type=int, default=5555,
                        help='port number for C-S communication')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    server = ServerTask(args)
    server.start()
    server.join()
