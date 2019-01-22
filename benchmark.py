import random
import sys
import threading
import time
from copy import deepcopy

from bert_serving.client import BertClient
from bert_serving.server import BertServer, get_args_parser
from bert_serving.server.helper import get_run_args
from numpy import mean


def tprint(msg):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def get_benchmark_parser():
    parser = get_args_parser()

    group = parser.add_argument_group('Benchmark parameters',
                                      'config the experiments of the benchmark')

    group.add_argument('-default_max_batch_size', type=int, default=512,
                       help='default value for maximum number of sequences handled by each worker')
    group.add_argument('-default_max_seq_len', type=int, default=32,
                       help='default value for maximum length of a sequence')
    group.add_argument('-default_num_client', type=int, default=1,
                       help='default value for number of concurrent clients')
    group.add_argument('-default_client_batch_size', type=int, default=4096,
                       help='default value for client batch size')
    group.add_argument('-default_pooling_layer', type=int, default=-2,
                       help='default value for pooling layer')

    group.add_argument('-test_client_batch_size', type=int, nargs='+', default=[1, 16, 256, 4096])
    group.add_argument('-test_max_batch_size', type=int, nargs='+', default=[8, 32, 128, 512])
    group.add_argument('-test_max_seq_len', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    group.add_argument('-test_num_client', type=int, nargs='+', default=[1, 4, 16, 64])
    group.add_argument('-test_pooling_layer', type=int, nargs='+', default=[[-j] for j in range(1, 13)])

    group.add_argument('-wait_till_ready', type=int, default=30,
                       help='seconds to wait until server is ready to serve')
    group.add_argument('-client_vocab_source', type=str, default='README.md',
                       help='file path for building client vocabulary')
    group.add_argument('-num_repeat', type=int, default=10,
                       help='number of repeats per experiment (must >2), '
                            'as the first two results are omitted for warm-up effect')
    return parser


args = get_run_args(get_benchmark_parser)

with open(args.client_vocab_source, encoding='utf8') as fp:
    vocab = list(set(vv for v in fp for vv in v.strip().split()))
tprint('vocabulary size: %d' % len(vocab))


class BenchmarkClient(threading.Thread):
    def __init__(self):
        super().__init__(cargs)
        self.batch = [' '.join(random.choices(vocab, k=cargs.max_seq_len)) for _ in range(cargs.client_batch_size)]
        self.num_repeat = cargs.num_repeat
        self.avg_time = 0

    def run(self):
        with BertClient(port=cargs.port, port_out=cargs.port_out,
                        show_server_config=False, check_version=False, check_length=False) as bc:
            time_all = []
            for _ in range(self.num_repeat):
                start_t = time.perf_counter()
                bc.encode(self.batch)
                time_all.append(time.perf_counter() - start_t)
            self.avg_time = mean(time_all[2:])  # first one is often slow due to cold-start/warm-up effect


if __name__ == '__main__':
    experiments = ['client_batch_size', 'max_batch_size', 'max_seq_len', 'num_client', 'pooling_layer']
    fp = open('benchmark-%d%s.result' % (args.num_worker, '-fp16' if args.fp16 else ''), 'w')
    for exp_name in experiments:
        # set common args
        cargs = deepcopy(args)
        # set default value
        for v in experiments:
            setattr(cargs, v, vars(args)['default_%s' % v])
        exp_vars = vars(args)['test_%s' % exp_name]
        avg_speed = []
        fp.write('speed wrt. %s\n' % exp_name)
        for cvar in exp_vars:
            # override exp args
            setattr(cargs, exp_name, cvar)
            server = BertServer(cargs)
            server.start()
            time.sleep(cargs.wait_till_ready)

            # sleep until server is ready
            all_clients = [BenchmarkClient(cargs) for _ in range(cargs.num_client)]
            for bc in all_clients:
                bc.start()

            clients_speed = []
            for bc in all_clients:
                bc.join()
                clients_speed.append(cargs.client_batch_size / bc.avg_time)
            server.close()

            max_speed, min_speed, cavg_speed = int(max(clients_speed)), int(min(clients_speed)), int(
                mean(clients_speed))

            tprint('avg speed: %d\tmax speed: %d\tmin speed: %d' % (cavg_speed, max_speed, min_speed))
            fp.write('%s\t%d\n' % (cvar, cavg_speed))
            fp.flush()
            avg_speed.append(cavg_speed)

        # for plotting
        fp.write('%s\n%s\n' % (exp_vars, avg_speed))
        fp.flush()
    fp.close()
