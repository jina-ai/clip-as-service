import tensorflow as tf

from plugin.quantizer.nn import get_var


class BaseQuantizer:
    def __init__(self, bits=4, learning_rate=1e-3, init_min_val=-12, init_max_val=20):
        num_centroids = 2 ** bits
        self.centroids = get_var('centroids', [num_centroids],
                                 initializer_fn=tf.initializers.random_uniform(init_min_val, init_max_val))
        self.ph_x = tf.placeholder(tf.float32, shape=[None, None], name='original_input')
        self.batch_size = tf.shape(self.ph_x)[0]
        self.num_dim = tf.shape(self.ph_x)[1]
        tiled_x = tf.tile(tf.expand_dims(self.ph_x, axis=1), [1, num_centroids, 1])
        tiled_centroids = tf.tile(tf.expand_dims(tf.expand_dims(self.centroids, axis=0), axis=2),
                                  [self.batch_size, 1, self.num_dim])

        dist = tf.abs(tiled_x - tiled_centroids)
        self.quant_x = tf.argmin(dist, axis=1, output_type=tf.int32)
        self.recover_x = tf.nn.embedding_lookup(self.centroids, self.quant_x)
        self.loss = tf.reduce_mean(tf.squared_difference(self.ph_x, self.recover_x))
        recover_dist = tf.abs(self.ph_x - self.recover_x)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-7)
        self.train_op = optimizer.minimize(self.loss)

        self.statistic = {
            'x_m': tf.reduce_min(self.ph_x),
            'x_M': tf.reduce_max(self.ph_x),
            'x_a': tf.reduce_mean(self.ph_x),
            'rx_m': tf.reduce_min(self.recover_x),
            'rx_M': tf.reduce_max(self.recover_x),
            'rx_a': tf.reduce_mean(self.recover_x),
            '|D|': tf.size(tf.unique(tf.reshape(self.quant_x, [-1]))[0]),
            'd_m': tf.reduce_min(recover_dist),
            'd_M': tf.reduce_max(recover_dist),
            'd_a': tf.reduce_mean(recover_dist)
        }
