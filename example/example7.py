import time
from collections import namedtuple

import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.bert.extract_features import PoolingStrategy
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

data = pd.read_csv('/data/cips/data/lab/data/dataset/uci-news-aggregator.csv', usecols=['TITLE', 'CATEGORY'])

# I do aspire here to have balanced classes
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
print('min_seq_len: %d' % min(len(v.split()) for v in subset_text))
print('max_seq_len: %d' % max(len(v.split()) for v in subset_text))
print('unique label: %d' % num_label)

pool_layer = 1
subset_vec_all_layers = []

common = {
    'model_dir': '//data/cips/data/lab/data/model/uncased_L-12_H-768_A-12',
    'num_worker': 2,
    'num_repeat': 5,
    'port': 6006,
    'port_out': 6007,
    'max_seq_len': 20,
    'client_batch_size': 2048,
    'max_batch_size': 256,
    'num_client': 1,
    'pooling_strategy': PoolingStrategy.REDUCE_MEAN,
    'pooling_layer': [-2],
    'gpu_memory_fraction': 0.5
}
args = namedtuple('args_namedtuple', ','.join(common.keys()))
for k, v in common.items():
    setattr(args, k, v)

for pool_layer in range(1, 13):
    setattr(args, 'pooling_layer', [-pool_layer])
    server = BertServer(args)
    server.start()
    print('wait until server is ready...')
    time.sleep(15)
    print('encoding...')
    bc = BertClient(port=common['port'], port_out=common['port_out'], show_server_config=True)
    subset_vec_all_layers.append(bc.encode(subset_text))
    bc.close()
    server.close()
    print('done at layer -%d' % pool_layer)


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
    fig.suptitle('%s visualization of BERT layers using "bert-as-service" (-pool_strategy=%s)' % (vis_alg, pool_alg),
                 fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.82, 0.1, 0.01, 0.3])
    cbar = plt.colorbar(cax=cax, ticks=range(num_label))
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['ent.', 'bus.', 'sci.', 'heal.']):
        cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', rotation=270)
    plt.show()


pca_embed = [PCA(n_components=2).fit_transform(v) for v in subset_vec_all_layers]
vis(pca_embed)

if False:
    tsne_embed = [TSNE(n_jobs=8).fit_transform(v) for v in subset_vec_all_layers]
    vis(tsne_embed, 't-SNE')
