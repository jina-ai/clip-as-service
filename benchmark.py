import random
import string
import sys
import threading
import time
from collections import namedtuple

from numpy import mean

from service.client import BertClient
from service.server import BertServer

PORT = 5557


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
        bc = BertClient(port=PORT)
        for _ in range(self.num_repeat):
            start_t = time.perf_counter()
            bc.encode(self.batch)
            time_all.append(time.perf_counter() - start_t)
        print(time_all)
        self.avg_time = mean(time_all)


if __name__ == '__main__':
    common = {
        'model_dir': '/data/cips/data/lab/data/model/chinese_L-12_H-768_A-12',
        'num_worker': 4,
        'num_repeat': 3,
        'port': PORT,
        'max_seq_len': 40,
        'client_batch_size': 2048,
        'max_batch_size': 256,
        'num_client': 1
    }
    experiments = {
        'num_client': [2, 4, 8, 16, 32],
        'max_batch_size': [32, 64, 128, 256, 512],
        'max_seq_len': [20, 40, 80, 160, 320],
        'client_batch_size': [256, 512, 1024, 2048, 4096],
    }

    for var_name, var_lst in experiments.items():
        # set common args
        args = namedtuple('args', ','.join(common.keys()))
        for k, v in common.items():
            setattr(args, k, v)

        avg_speed = []
        for var in var_lst:
            # override exp args
            setattr(args, var_name, var)
            server = BertServer(args)
            server.start()

            # sleep until server is ready
            time.sleep(15)
            all_clients = [BenchmarkClient(args) for _ in range(args.num_client)]

            tprint('num_client: %d' % len(all_clients))
            for bc in all_clients:
                bc.start()

            all_thread_speed = []
            for bc in all_clients:
                bc.join()
                cur_speed = args.client_batch_size / bc.avg_time
                all_thread_speed.append(cur_speed)

            max_speed = int(max(all_thread_speed))
            min_speed = int(min(all_thread_speed))
            t_avg_speed = int(mean(all_thread_speed))

            tprint('%s: %5d\t%.3f\t%d/s' % (var_name, var, bc.avg_time, t_avg_speed))
            tprint('max speed: %d\t min speed: %d' % (max_speed, min_speed))
            avg_speed.append(t_avg_speed)
            server.close()
        tprint('______\nspeed wrt. %s' % var_name)
        for i, j in zip(var_lst, avg_speed):
            tprint('%d\t%d' % (i, j))
        tprint('______')
