import random
import string
import sys
import threading
import time
from collections import namedtuple

from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.graph import PoolingStrategy
from numpy import mean

PORT = 7779
PORT_OUT = 7780

common = {
    'model_dir': '/data/cips/save/chinese_L-12_H-768_A-12',
    'num_worker': 2,
    'num_repeat': 5,
    'port': PORT,
    'port_out': PORT_OUT,
    'max_seq_len': 40,
    'client_batch_size': 2048,
    'max_batch_size': 256,
    'num_client': 1,
    'pooling_strategy': PoolingStrategy.REDUCE_MEAN,
    'pooling_layer': [-2],
    'gpu_memory_fraction': 0.5,
    'prefetch_size': 10,
    'xla': False,
    'cpu': False,
    'verbose': False,
    'config_name': 'bert_config.json',
    'ckpt_name': 'bert_model.ckpt',
    'tuned_model_dir': None,
    'mask_cls_sep': False,
}

args = namedtuple('args_nt', ','.join(common.keys()))
globals()[args.__name__] = args


def tprint(msg):
    """like print, but won't get newlines confused with multiple threads"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


class BenchmarkClient(threading.Thread):
    def __init__(self):
        super().__init__()
        self.batch = [''.join(random.choices(string.ascii_uppercase + string.digits,
                                             k=args.max_seq_len)) for _ in range(args.client_batch_size)]

        self.num_repeat = args.num_repeat
        self.avg_time = 0

    def run(self):
        time_all = []
        bc = BertClient(port=PORT, port_out=PORT_OUT, show_server_config=False, check_version=False)
        for _ in range(self.num_repeat):
            start_t = time.perf_counter()
            bc.encode(self.batch)
            time_all.append(time.perf_counter() - start_t)
        print(time_all)
        self.avg_time = mean(time_all)


if __name__ == '__main__':

    experiments = {
        'client_batch_size': [1, 4, 8, 16, 64, 256, 512, 1024, 2048, 4096],
        'max_batch_size': [32, 64, 128, 256, 512],
        'max_seq_len': [20, 40, 80, 160, 320],
        'num_client': [2, 4, 8, 16, 32],
        'pooling_layer': [[-j] for j in range(1, 13)]
    }

    fp = open('benchmark-%d.result' % common['num_worker'], 'w')
    for var_name, var_lst in experiments.items():
        # set common args
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
            all_clients = [BenchmarkClient() for _ in range(args.num_client)]

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

            tprint('%s: %s\t%.3f\t%d/s' % (var_name, var, bc.avg_time, t_avg_speed))
            tprint('max speed: %d\t min speed: %d' % (max_speed, min_speed))
            avg_speed.append(t_avg_speed)
            server.close()

        fp.write('#### Speed wrt. `%s`\n\n' % var_name)
        fp.write('|`%s`|seqs/s|\n' % var_name)
        fp.write('|---|---|\n')
        for i, j in zip(var_lst, avg_speed):
            fp.write('|%s|%d|\n' % (i, j))
            fp.flush()
    fp.close()
