import json
import os

import GPUtil
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable())

train_fp = ['/data/cips/data/lab/data/dataset/final_all_data/exercise_contest/data_train.json']
batch_size = 256


def get_encodes(x):
    samples = [json.loads(l) for l in x]
    text = [s['fact'][-50:] for s in samples]
    return json.dumps(text).encode()


data_node = (tf.data.TextLineDataset(train_fp).batch(batch_size)
             .map(lambda x: tf.py_func(get_encodes, [x], tf.string, name='train_mktokens_fn'),
                  num_parallel_calls=4)
             .make_one_shot_iterator().get_next())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        print(sess.run(data_node))
