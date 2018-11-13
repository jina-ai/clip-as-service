import random
import string
import sys
import threading
import time
from collections import namedtuple

from numpy import mean

from service.client import BertClient
from service.server import BertServer


def tprint(msg):
    """like print, but won't get newlines confused with multiple threads"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


class BenchmarkClient(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.batch = [''.join(random.choices(string.ascii_uppercase + string.digits,
                                             k=args.max_seq_len)) for _ in range(args.client_batch_size)]

        self.bc = BertClient()
        self.num_repeat = args.num_repeat
        self.avg_time = 0

    def run(self):
        time_all = []
        for _ in range(self.num_repeat):
            start_t = time.perf_counter()
            self.bc.encode(self.batch)
            time_all.append(time.perf_counter() - start_t)
        print(time_all)
        self.avg_time = mean(time_all)


if __name__ == '__main__':
    common = {
        'model_dir': '/data/cips/save/chinese-bert/chinese_L-12_H-768_A-12/',
        'num_worker': 8,
        'num_repeat': 10,
        'port': 5555
    }
    experiments = [
        {
            'max_seq_len': [20, 40, 80, 160],
            'batch_size_per_worker': 128,
            'client_batch_size': 2048,
            'client_seq_len': 100,
            'num_client': 1
        },
        {
            'max_seq_len': 20,
            'batch_size_per_worker': 128,
            'client_batch_size': [256, 1024, 2048, 4096],
            'client_seq_len': 100,
            'num_client': 1,
        },
        {
            'max_seq_len': 20,
            'batch_size_per_worker': 128,
            'client_batch_size': 2048,
            'client_seq_len': 100,
            'num_client': [1, 2, 3, 4],
        },
    ]

    for exp, var_name in zip(experiments, ['max_seq_len', 'client_batch_size', 'num_client']):
        avg_speed = []
        for var in exp[var_name]:
            args = namedtuple('args', ','.join(list(common.keys()) + list(exp.keys())))
            for k, v in common.items():
                setattr(args, k, v)
            for k, v in exp.items():
                setattr(args, k, v)
            # override the var_name
            setattr(args, var_name, var)

            server = BertServer(args)
            server.start()

            # sleep until server is ready
            time.sleep(15)
            all_clients = []
            for _ in range(args.num_client):
                bc = BenchmarkClient(args)
                bc.start()
                all_clients.append(bc)

            for bc in all_clients:
                bc.join()
                cur_speed = args.client_batch_size / bc.avg_time
                tprint('%s: %5d\t%.3f\t%d/s' % (var_name, var, bc.avg_time, int(cur_speed)))
                avg_speed.append(cur_speed)
            server.close()
        tprint('______\nspeed wrt. %s' % var_name)
        for i, j in zip(exp[var_name], avg_speed):
            tprint('%d\t%d' % (i, j))
        tprint('______')
