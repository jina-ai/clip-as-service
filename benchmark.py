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

        self.num_repeat = args.num_repeat
        self.avg_time = 0

    def run(self):
        time_all = []
        bc = BertClient()
        for _ in range(self.num_repeat):
            start_t = time.perf_counter()
            bc.encode(self.batch)
            time_all.append(time.perf_counter() - start_t)
        print(time_all)
        self.avg_time = mean(time_all)


if __name__ == '__main__':
    common = {
        'model_dir': '/data/cips/result/chinese_L-12_H-768_A-12/',
        'num_worker': 4,
        'num_repeat': 10,
        'port': 5555
    }
    experiments = [
        # {
        #     'max_seq_len': 40,
        #     'max_batch_size': [32, 64, 128, 256],
        #     'client_batch_size': 2048,
        #     'num_client': 1
        # },
        # {
        #     'max_seq_len': [20, 40, 80, 160],
        #     'max_batch_size': 128,
        #     'client_batch_size': 2048,
        #     'num_client': 1
        # },
        # {
        #     'max_seq_len': 40,
        #     'max_batch_size': 128,
        #     'client_batch_size': [256, 1024, 2048, 4096],
        #     'num_client': 1,
        # },
        {
            'max_seq_len': 40,
            'max_batch_size': 128,
            'client_batch_size': 2048,
            'num_client': [2, 4, 8, 16],
        },
    ]

    for cur_exp in experiments:
        var_name = [k for k, v in cur_exp.items() if isinstance(v, list)][0]
        avg_speed = []
        for var in cur_exp[var_name]:
            args = namedtuple('args', ','.join(list(common.keys()) + list(cur_exp.keys())))
            for k, v in common.items():
                setattr(args, k, v)
            for k, v in cur_exp.items():
                setattr(args, k, v)
            # override the var_name
            setattr(args, var_name, var)

            server = BertServer(args)
            server.start()

            # sleep until server is ready
            time.sleep(15)
            all_clients = [BenchmarkClient(args) for _ in range(args.num_client)]
            tprint('num_client: %d' % len(all_clients))
            for bc in all_clients:
                bc.start()
                print('start one')

            for bc in all_clients:
                bc.join()

            cur_speed = args.client_batch_size / bc.avg_time
            tprint('%s: %5d\t%.3f\t%d/s' % (var_name, var, bc.avg_time, int(cur_speed)))
            avg_speed.append(cur_speed)
            server.close()
        tprint('______\nspeed wrt. %s' % var_name)
        for i, j in zip(cur_exp[var_name], avg_speed):
            tprint('%d\t%d' % (i, j))
        tprint('______')
