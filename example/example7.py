import random

import matplotlib

matplotlib.use('Agg')
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
from sklearn.decomposition import PCA

from bert_serving.client import BertClient

num_sample = 10000

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

bc = BertClient(port=6000, port_out=6001)
subset_vec = bc.encode(subset_text)
# embeddings = TSNE(n_jobs=8).fit_transform(subset_vec)
pca = PCA(n_components=2)
embeddings = pca.fit_transform(subset_vec)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
plt.scatter(vis_x, vis_y, c=subset_label, cmap=plt.cm.get_cmap("jet", num_label), marker='.')
plt.colorbar(ticks=range(num_label))
savefig('layer-%d.png' % random.randint(0, 100), bbox_inches='tight')
