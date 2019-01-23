import random
import threading
import time

from numpy import mean


class BenchmarkClient(threading.Thread):
    def __init__(self, cargs, vocab):
        super().__init__()
        self.batch = [' '.join(random.choices(vocab, k=cargs.max_seq_len)) for _ in range(cargs.client_batch_size)]
        self.num_repeat = cargs.num_repeat
        self.avg_time = 0
        self.port = cargs.port
        self.port_out = cargs.port_out

    def run(self):
        try:
            from bert_serving.client import BertClient
        except ImportError:
            raise ImportError('BertClient module is not available, it is required for benchmarking.'
                              'Please use "pip install -U bert-serving-client" to install it.')
        with BertClient(port=self.port, port_out=self.port_out,
                        show_server_config=True, check_version=False, check_length=False) as bc:
            time_all = []
            for _ in range(self.num_repeat):
                start_t = time.perf_counter()
                bc.encode(self.batch)
                time_all.append(time.perf_counter() - start_t)
            self.avg_time = mean(time_all[2:])  # first one is often slow due to cold-start/warm-up effect


def run_benchmark(args):
    from copy import deepcopy
    from bert_serving.server import BertServer

    # load vocabulary
    with open(args.client_vocab_file, encoding='utf8') as fp:
        vocab = list(set(vv for v in fp for vv in v.strip().split()))
    print('vocabulary size: %d' % len(vocab))

    # select those non-empty test cases
    all_exp_names = [k.replace('test_', '') for k, v in vars(args).items() if k.startswith('test_') and v]

    for exp_name in all_exp_names:
        # set common args
        cargs = deepcopy(args)
        exp_vars = vars(args)['test_%s' % exp_name]
        avg_speed = []

        for cvar in exp_vars:
            # override exp args
            setattr(cargs, exp_name, cvar)
            server = BertServer(cargs)
            server.start()
            time.sleep(cargs.wait_till_ready)

            # sleep until server is ready
            all_clients = [BenchmarkClient(cargs, vocab) for _ in range(cargs.num_client)]
            for bc in all_clients:
                bc.start()

            clients_speed = []
            for bc in all_clients:
                bc.join()
                clients_speed.append(cargs.client_batch_size / bc.avg_time)
            server.close()

            max_speed, min_speed, cavg_speed = int(max(clients_speed)), int(min(clients_speed)), int(
                mean(clients_speed))

            print('avg speed: %d\tmax speed: %d\tmin speed: %d' % (cavg_speed, max_speed, min_speed), flush=True)

            avg_speed.append(cavg_speed)

        with open('benchmark-%d%s.result' % (args.num_worker, '-fp16' if args.fp16 else ''), 'a') as fw:
            print('\n|`%s`\t|samples/s|\n|---|---|' % exp_name, file=fw)
            for cvar, cavg_speed in zip(exp_vars, avg_speed):
                print('|%s\t|%d|' % (cvar, cavg_speed), file=fw)
            # for additional plotting
            print('\n%s = %s\n%s = %s' % (exp_name, exp_vars, 'speed', avg_speed), file=fw)
