#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# visualizing a 12-layer BERT

import time
from collections import namedtuple

import numpy as np
import pandas as pd
# from MulticoreTSNE import MulticoreTSNE as TSNE
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


#=========================== dump bert vectors ===========================
data = pd.read_csv('/corpus/uci-news-aggregator.csv', usecols=['TITLE', 'CATEGORY'])

# just copy paste from some Kaggle kernel ->
num_of_categories = 5000
shuffled = data.reindex(np.random.permutation(data.index))
e = shuffled[shuffled['CATEGORY'] == 'e'][:num_of_categories]
b = shuffled[shuffled['CATEGORY'] == 'b'][:num_of_categories]
t = shuffled[shuffled['CATEGORY'] == 't'][:num_of_categories]
m = shuffled[shuffled['CATEGORY'] == 'm'][:num_of_categories]
concated = pd.concat([e, b, t, m], ignore_index=True)
# Shuffle the dataset
concated = concated.reindex(np.random.permutation(concated.index))
concated['LABEL'] = 0
# One-hot encode the lab
concated.loc[concated['CATEGORY'] == 'e', 'LABEL'] = 0
concated.loc[concated['CATEGORY'] == 'b', 'LABEL'] = 1
concated.loc[concated['CATEGORY'] == 't', 'LABEL'] = 2
concated.loc[concated['CATEGORY'] == 'm', 'LABEL'] = 3

subset_text = list(concated['TITLE'].values)
subset_label = list(concated['LABEL'].values)
num_label = len(set(subset_label))

# <- just copy paste from some Kaggle kernel

print('min_seq_len: %d' % min(len(v.split()) for v in subset_text))
print('max_seq_len: %d' % max(len(v.split()) for v in subset_text))
print('unique label: %d' % num_label)

pool_layer = 1
subset_vec_all_layers = []
port = 6006
port_out = 6007

common = [
    '-model_dir', '/bert_model/chinese_L-12_H-768_A-12/',
    '-num_worker', '2',
    '-port', str(port),
    '-port_out', str(port_out),
    '-max_seq_len', '20',
    # '-client_batch_size', '2048',
    '-max_batch_size', '256',
    # '-num_client', '1',
    '-pooling_strategy', 'REDUCE_MEAN',
    '-pooling_layer', '-2',
    '-gpu_memory_fraction', '0.2',
    '-device','3',
]
args = get_args_parser().parse_args(common)

for pool_layer in range(1, 13):
    setattr(args, 'pooling_layer', [-pool_layer])
    server = BertServer(args)
    server.start()
    print('wait until server is ready...')
    time.sleep(20)
    print('encoding...')
    bc = BertClient(port=port, port_out=port_out, show_server_config=True)
    subset_vec_all_layers.append(bc.encode(subset_text))
    bc.close()
    server.close()
    print('done at layer -%d' % pool_layer)

#save bert vectors and labels
stacked_subset_vec_all_layers = np.stack(subset_vec_all_layers)
np.save('example7_5k_2',stacked_subset_vec_all_layers)
np_subset_label = np.array(subset_label)
np.save('example7_5k_2_subset_label',np_subset_label)

#load bert vectors and labels
subset_vec_all_layers = np.load('example7_5k_mxnet.npy')
np_subset_label = np.load('example7_5k_mxnet_subset_label.npy')
subset_label = np_subset_label.tolist()
#=========================== visualize ===========================
def vis(embed, vis_alg='PCA', pool_alg='REDUCE_MEAN'):
    plt.close()
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [21, 7]
    for idx, ebd in enumerate(embed):
        ax = plt.subplot(2, 6, idx + 1)
        vis_x = ebd[:, 0]
        vis_y = ebd[:, 1]
        plt.scatter(vis_x, vis_y, c=subset_label, cmap=ListedColormap(["blue", "green", "yellow", "red"]), marker='.',
                    alpha=0.7, s=2)
        ax.set_title('pool_layer=-%d' % (idx + 1))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.95, top=0.9)
    cax = plt.axes([0.96, 0.1, 0.01, 0.3])
    cbar = plt.colorbar(cax=cax, ticks=range(num_label))
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['ent.', 'bus.', 'sci.', 'heal.']):
        cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', rotation=270)
    fig.suptitle('%s visualization of BERT layers using "bert-as-service" (-pool_strategy=%s)' % (vis_alg, pool_alg),
                 fontsize=14)
    plt.show()


pca_embed = [PCA(n_components=2).fit_transform(v) for v in subset_vec_all_layers]
vis(pca_embed)

# if False:
#     tsne_embed = [TSNE(n_jobs=8).fit_transform(v) for v in subset_vec_all_layers]
#     vis(tsne_embed, 't-SNE')
