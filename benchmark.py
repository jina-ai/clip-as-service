import argparse
import random
import sys
import threading
import time
from collections import namedtuple

from bert_serving.client import BertClient
from bert_serving.server import BertServer, get_args_parser
from bert_serving.server.helper import get_run_args
from numpy import mean


def tprint(msg):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def get_benchmark_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', type=str, required=True,
                        help='BERT model dir')
    parser.add_argument('-port', type=int, default=7779,
                        help='port of the bert server')
    parser.add_argument('-port_out', type=int, default=7780,
                        help='output port of the bert server')
    parser.add_argument('-num_worker', type=int, default=1,
                        help='number of workers')
    parser.add_argument('-num_repeat', type=int, default=10,
                        help='number of repeats per experiment (must >2), '
                             'as the first two results are omitted for warm-up effect')
    parser.add_argument('-fp16', action='store_true', default=False,
                        help='use float16 precision (experimental)')

    parser.add_argument('-default_max_batch_size', type=int, default=512,
                        help='default value for maximum number of sequences handled by each worker')
    parser.add_argument('-default_max_seq_len', type=int, default=32,
                        help='default value for maximum length of a sequence')
    parser.add_argument('-default_num_client', type=int, default=1,
                        help='default value for number of concurrent clients')
    parser.add_argument('-default_client_batch_size', type=int, default=4096,
                        help='default value for client batch size')
    parser.add_argument('-test_client_batch_size', type=int, nargs='+', default=[1, 16, 256, 4096])
    parser.add_argument('-test_max_batch_size', type=int, nargs='+', default=[8, 32, 128, 512])
    parser.add_argument('-test_max_seq_len', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    parser.add_argument('-test_num_client', type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('-test_pooling_layer', type=int, nargs='+', default=[[-j] for j in range(1, 13)])

    parser.add_argument('-wait_till_ready', type=int, default=30,
                        help='seconds to wait until server is ready to serve')
    parser.add_argument('-client_vocab_source', type=str, default='README.md',
                        help='file path for building client vocabulary')
    return parser


common = vars(get_args_parser().parse_args(['-model_dir', '']))
for k, v in vars(get_run_args(get_benchmark_parser)).items():
    common[k] = v

param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(common.items())])
tprint('%20s   %s\n%s\n%s\n' % ('ARG', 'VALUE', '_' * 50, param_str))

with open(common['client_vocab_source'], encoding='utf8') as fp:
    vocab = list(set(vv for v in fp for vv in v.strip().split()))
tprint('vocabulary size: %d' % len(vocab))

args = namedtuple('args_nt', ','.join(common.keys()))
globals()[args.__name__] = args


class BenchmarkClient(threading.Thread):
    def __init__(self):
        super().__init__()
        self.batch = [' '.join(random.choices(vocab, k=args.max_seq_len)) for _ in range(args.client_batch_size)]
        self.num_repeat = args.num_repeat
        self.avg_time = 0

    def run(self):
        with BertClient(port=args.port, port_out=args.port_out,
                        show_server_config=False, check_version=False, check_length=False) as bc:
            time_all = []
            for _ in range(self.num_repeat):
                start_t = time.perf_counter()
                bc.encode(self.batch)
                time_all.append(time.perf_counter() - start_t)
            self.avg_time = mean(time_all[2:])  # first one is often slow due to cold-start/warm-up effect


if __name__ == '__main__':
    experiments = {k: common['test_%s' % k] for k in
                   ['client_batch_size', 'max_batch_size', 'max_seq_len', 'num_client', 'pooling_layer']}

    fp = open('benchmark-%d%s.result' % (args.num_worker, '-fp16' if args.fp16 else ''), 'w')
    for var_name, var_lst in experiments.items():
        # set common args
        for k, v in common.items():
            setattr(args, k, v)

        avg_speed = []
        fp.write('speed wrt. %s\n' % var_name)
        for cvar in var_lst:
            # override exp args
            setattr(args, var_name, cvar)
            server = BertServer(args)
            server.start()
            time.sleep(args.wait_till_ready)

            # sleep until server is ready
            all_clients = [BenchmarkClient() for _ in range(args.num_client)]
            for bc in all_clients:
                bc.start()

            clients_speed = []
            for bc in all_clients:
                bc.join()
                clients_speed.append(args.client_batch_size / bc.avg_time)
            server.close()

            max_speed, min_speed, cavg_speed = int(max(clients_speed)), int(min(clients_speed)), int(
                mean(clients_speed))

            tprint('avg speed: %d\tmax speed: %d\tmin speed: %d' % (cavg_speed, max_speed, min_speed))
            fp.write('%s\t%d\n' % (cvar, cavg_speed))
            fp.flush()
            avg_speed.append(cavg_speed)

        # for plotting
        fp.write('%s\n%s\n' % (var_lst, avg_speed))
        fp.flush()
    fp.close()
