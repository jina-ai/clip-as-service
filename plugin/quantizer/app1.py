import json
import os
import random

import tensorflow as tf
from bert_serving.client import ConcurrentBertClient
from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks

from plugin.quantizer.base_quantizer import PiecewiseQuantizer

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

train_fp = ['/data/cips/data/larry-autoencoder/cail_0518/data_train.json']
dev_fp = ['/data/cips/data/larry-autoencoder/cail_0518/data_valid.json']
num_parallel_calls = 4
num_bits = 4

bc = ConcurrentBertClient(port=5500, port_out=5501)


def get_encodes(x, shuffle=False):
    # x is `batch_size` of lines, each of which is a json object
    samples = [json.loads(l) for l in x]
    texts = []
    for s in samples:
        t = s['fact']
        s_idx = random.randint(0, len(t) - 1) if shuffle else 0
        texts.append(t[s_idx: (s_idx + 40)])
    features = bc.encode(texts)
    return features


def get_ds(fp, batch_size=1024, shuffle=False, only_head=False):
    _get_encodes = lambda x: get_encodes(x, shuffle)
    ds = (tf.data.TextLineDataset(fp).batch(batch_size)
          .map(lambda x: tf.py_func(_get_encodes, [x], tf.float32, name='bert_client'),
               num_parallel_calls=num_parallel_calls))
    if shuffle:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(5))
    if only_head:
        ds = ds.take(1).repeat(-1)
    return ds.prefetch(5).make_one_shot_iterator().get_next()


def get_config():
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement = False
    return config


def plot_graph(font_size=16):
    plt.close()
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [21, 16]
    plt.plot(hist_centroids, 'ro-', markersize=10, linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.xlabel('iterations', fontsize=font_size)
    plt.ylabel('feature value', fontsize=font_size)
    plt.grid(True)

    xticks(range(len(hist_iters)), [str(v) for v in hist_iters], fontsize=font_size)
    plt.title('Quantization with %d bits, i.e. %d centroids' % (num_bits, 2 ** num_bits), fontsize=font_size)
    plt.show()


quantizer = PiecewiseQuantizer(dim_per_byte=4, num_dim=768)

sess = tf.Session(config=get_config())
sess.run(tf.global_variables_initializer())
iter, iter_per_save = 0, 100

train_ds = get_ds(train_fp, shuffle=True)
dev_ds = get_ds(dev_fp)
dev_x = sess.run(dev_ds)

hist_val_loss = []
hist_iters = []

while True:
    if iter % iter_per_save == 0:
        loss, stat = sess.run([quantizer.loss, quantizer.statistic],
                              feed_dict={quantizer.ph_x: dev_x})
        stat_str = ' '.join('%5s %.3f' % (k, v) for k, v in sorted(stat.items()))
        print('[V]%5d: %.5f %s' % (iter, loss, stat_str))
        hist_val_loss.append(loss)
        hist_iters.append(iter)
    x = sess.run(train_ds)
    loss, stat, _ = sess.run([quantizer.loss, quantizer.statistic, quantizer.train_op],
                             feed_dict={quantizer.ph_x: x})
    iter += 1
    stat_str = ' '.join('%5s %.3f' % (k, v) for k, v in sorted(stat.items()))
    print('[T]%5d: %.5f %s' % (iter, loss, stat_str))
