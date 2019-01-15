import tensorflow as tf

from .nn import get_var


class BaseQuantizer:
    def __init__(self, num_centroids=100, learning_rate=1e-3):
        self.centroids = get_var('centroids', [num_centroids])
        self.ph_x = tf.placeholder(tf.float32, shape=[None, None], name='original_input')
        self.batch_size, self.num_dim = tf.shape(self.ph_x)
        x = tf.tile(tf.expand_dims(self.ph_x, axis=1), [1, num_centroids, 1])
        centroids = tf.tile(tf.expand_dims(tf.expand_dims(self.centroids, axis=0), axis=2),
                            [self.batch_size, 1, self.num_dim])

        dist = tf.abs(x - centroids)
        self.quant_x = tf.argmin(dist, axis=1)
        self.recover_x = tf.nn.embedding_lookup(centroids, self.quant_x)
        self.loss = tf.reduce_mean(tf.squared_difference(self.quant_x, self.recover_x))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-7)
        self.train_op = optimizer.minimize(self.loss)
