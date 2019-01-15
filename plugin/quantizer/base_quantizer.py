import tensorflow as tf

from plugin.quantizer.nn import get_var


class BaseQuantizer:
    def __init__(self, num_centroids=100, learning_rate=1e-3):
        self.centroids = get_var('centroids', [num_centroids])
        self.ph_x = tf.placeholder(tf.float32, shape=[None, None], name='original_input')
        self.batch_size = tf.shape(self.ph_x)[0]
        self.num_dim = tf.shape(self.ph_x)[1]
        tiled_x = tf.tile(tf.expand_dims(self.ph_x, axis=1), [1, num_centroids, 1])
        tiled_centroids = tf.tile(tf.expand_dims(tf.expand_dims(self.centroids, axis=0), axis=2),
                                  [self.batch_size, 1, self.num_dim])

        dist = tf.abs(tiled_x - tiled_centroids)
        # self.dist_shape = tf.shape(dist)
        self.quant_x = tf.argmin(dist, axis=1, output_type=tf.int32)
        self.recover_x = tf.nn.embedding_lookup(self.centroids, self.quant_x)
        # self.quant_x_shape = tf.shape(self.recover_x)
        self.loss = tf.reduce_mean(tf.squared_difference(self.ph_x, self.recover_x))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-7)
        self.train_op = optimizer.minimize(self.loss)

        self.statistic = {
            'x_min': tf.reduce_min(self.ph_x),
            'x_max': tf.reduce_max(self.ph_x),
            'x_mean': tf.reduce_mean(self.ph_x),
            'qx_min': tf.reduce_min(self.recover_x),
            'qx_max': tf.reduce_max(self.recover_x),
            'qx_mean': tf.reduce_mean(self.recover_x),
            'uniq_q': tf.size(tf.unique(tf.reshape(self.quant_x, [-1]))[0])
        }
