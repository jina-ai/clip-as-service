#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>


import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                  mode='FAN_IN',
                                                                  uniform=False,
                                                                  dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def minus_mask(x, mask, offset=1e30):
    """
    masking by subtract a very large number
    :param x: sequence data in the shape of [B, L, D]
    :param mask: 0-1 mask in the shape of [B, L]
    :param offset: very large negative number
    :return: masked x
    """
    return x - tf.expand_dims(1.0 - mask, axis=-1) * offset


def mul_mask(x, mask):
    """
    masking by multiply zero
    :param x: sequence data in the shape of [B, L, D]
    :param mask: 0-1 mask in the shape of [B, L]
    :return: masked x
    """
    return x * tf.expand_dims(mask, axis=-1)


def masked_reduce_mean(x, mask):
    return tf.reduce_sum(mul_mask(x, mask), axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)


def masked_reduce_max(x, mask):
    return tf.reduce_max(minus_mask(x, mask), axis=1)


def weighted_sparse_softmax_cross_entropy(labels, preds, weights):
    """
    computing sparse softmax cross entropy by weighting differently on classes
    :param labels: sparse label in the shape of [B], size of label is L
    :param preds: logit in the shape of [B, L]
    :param weights: weight in the shape of [L]
    :return: weighted sparse softmax cross entropy in the shape of [B]
    """

    return tf.losses.sparse_softmax_cross_entropy(labels,
                                                  logits=preds,
                                                  weights=get_bounded_class_weight(labels, weights))


def get_bounded_class_weight(labels, weights, ub=None):
    if weights is None:
        return 1.0
    else:
        w = tf.gather(weights, labels)
        w = w / tf.reduce_min(w)
        w = tf.clip_by_value(1.0 + tf.log1p(w),
                             clip_value_min=1.0,
                             clip_value_max=ub if ub is not None else tf.cast(tf.shape(weights)[0], tf.float32) / 2.0)
    return w


def weighted_smooth_softmax_cross_entropy(labels, num_labels, preds, weights,
                                          epsilon=0.1):
    """
        computing smoothed softmax cross entropy by weighting differently on classes
        :param epsilon: smoothing factor
        :param num_labels: maximum number of labels
        :param labels: sparse label in the shape of [B], size of label is L
        :param preds: logit in the shape of [B, L]
        :param weights: weight in the shape of [L]
        :return: weighted sparse softmax cross entropy in the shape of [B]
        """

    return tf.losses.softmax_cross_entropy(tf.one_hot(labels, num_labels),
                                           logits=preds,
                                           label_smoothing=epsilon,
                                           weights=get_bounded_class_weight(labels, weights))


def get_var(name, shape, dtype=tf.float32,
            initializer_fn=initializer,
            regularizer_fn=regularizer, **kwargs):
    return tf.get_variable(name, shape,
                           initializer=initializer_fn,
                           dtype=dtype,
                           regularizer=regularizer_fn, **kwargs)


def layer_norm(inputs,
               epsilon=1e-8,
               scope=None,
               reuse=None):
    """Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope or 'Layer_Normalize', reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

    return outputs


def linear_logit(x, units, act_fn=None, dropout_keep=1., use_layer_norm=False, scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'linear_logit', reuse=reuse):
        logit = tf.layers.dense(x, units=units, activation=act_fn,
                                kernel_initializer=initializer,
                                kernel_regularizer=regularizer)
        # do dropout
        logit = tf.nn.dropout(logit, keep_prob=dropout_keep)
        if use_layer_norm:
            logit = tf.contrib.layers.layer_norm(logit)
        return logit


def bilinear_logit(x, units, act_fn=None,
                   first_units=256,
                   first_act_fn=tf.nn.relu, scope=None, **kwargs):
    with tf.variable_scope(scope or 'bilinear_logit'):
        first = linear_logit(x, first_units, act_fn=first_act_fn, scope='first', **kwargs)
        return linear_logit(first, units, scope='second', act_fn=act_fn, **kwargs)


def label_smoothing(inputs, epsilon=0.1):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    """
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def normalize_by_axis(x, axis, smooth_factor=1e-5):
    x += smooth_factor
    return x / tf.reduce_sum(x, axis, keepdims=True)  # num A x num B


def get_cross_correlated_mat(num_out_A, num_out_B, learn_cooc='FIXED', cooc_AB=None, scope=None, reuse=None):
    with tf.variable_scope(scope or 'CrossCorrlated_Mat', reuse=reuse):
        if learn_cooc == 'FIXED' and cooc_AB is not None:
            pB_given_A = normalize_by_axis(cooc_AB, 1)
            pA_given_B = normalize_by_axis(cooc_AB, 0)
        elif learn_cooc == 'JOINT':
            share_cooc = tf.nn.relu(get_var('cooc_ab', shape=[num_out_A, num_out_B]))
            pB_given_A = normalize_by_axis(share_cooc, 1)
            pA_given_B = normalize_by_axis(share_cooc, 0)
        elif learn_cooc == 'DISJOINT':
            cooc1 = tf.nn.relu(get_var('pb_given_a', shape=[num_out_A, num_out_B]))
            cooc2 = tf.nn.relu(get_var('pa_given_b', shape=[num_out_A, num_out_B]))
            pB_given_A = normalize_by_axis(cooc1, 1)
            pA_given_B = normalize_by_axis(cooc2, 0)
        else:
            raise NotImplementedError

        return pA_given_B, pB_given_A


def get_self_correlated_mat(num_out_A, scope=None, reuse=None):
    with tf.variable_scope(scope or 'Self_Correlated_mat', reuse=reuse):
        cooc1 = get_var('pa_corr', shape=[num_out_A, num_out_A],
                        initializer_fn=tf.contrib.layers.variance_scaling_initializer(factor=0.1,
                                                                                      mode='FAN_AVG',
                                                                                      uniform=True,
                                                                                      dtype=tf.float32),
                        regularizer_fn=tf.contrib.layers.l2_regularizer(scale=3e-4))
        return tf.matmul(cooc1, cooc1, transpose_b=True) + tf.eye(num_out_A)


def gate_filter(x, scope=None, reuse=None):
    with tf.variable_scope(scope or 'Gate', reuse=reuse):
        threshold = get_var('threshold', shape=[])
        gate = tf.cast(tf.greater(x, threshold), tf.float32)
        return x * gate


from tensorflow.python.ops import array_ops


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def spatial_dropout(x, scope=None, reuse=None):
    input_dim = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope or 'spatial_dropout', reuse=reuse):
        d = tf.random_uniform(shape=[1], minval=0, maxval=input_dim, dtype=tf.int32)
        f = tf.one_hot(d, on_value=0., off_value=1., depth=input_dim)
        g = x * f  # do dropout
        g *= (1. + 1. / input_dim)  # do rescale
        return g


def get_last_output(output, seq_length, scope=None, reuse=None):
    """Get the last value of the returned output of an RNN.
    http://disq.us/p/1gjkgdr
    output: [batch x number of steps x ... ] Output of the dynamic lstm.
    sequence_length: [batch] Length of each of the sequence.
    """
    with tf.variable_scope(scope or 'gather_nd', reuse=reuse):
        rng = tf.range(0, tf.shape(seq_length)[0])
        indexes = tf.stack([rng, seq_length - 1], 1)
        return tf.gather_nd(output, indexes)


def get_lstm_init_state(batch_size, num_layers, num_units, direction, scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'lstm_init_state', reuse=reuse):
        num_dir = 2 if direction.startswith('bi') else 1
        c = get_var('lstm_init_c', shape=[num_layers * num_dir, num_units])
        c = tf.tile(tf.expand_dims(c, axis=1), [1, batch_size, 1])
        h = get_var('lstm_init_h', shape=[num_layers * num_dir, num_units])
        h = tf.tile(tf.expand_dims(h, axis=1), [1, batch_size, 1])
        return c, h


def dropout_res_layernorm(x, fx, act_fn=tf.nn.relu,
                          dropout_keep_rate=1.0,
                          residual=True,
                          normalize_output=True,
                          scope='rnd_block',
                          reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        input_dim = x.get_shape().as_list()[-1]
        output_dim = fx.get_shape().as_list()[-1]

        # do dropout
        fx = tf.nn.dropout(fx, keep_prob=dropout_keep_rate)

        if residual and input_dim != output_dim:
            res_x = tf.layers.conv1d(x,
                                     filters=output_dim,
                                     kernel_size=1,
                                     activation=None,
                                     name='res_1x1conv')
        else:
            res_x = x

        if residual and act_fn is None:
            output = fx + res_x
        elif residual and act_fn is not None:
            output = act_fn(fx + res_x)
        else:
            output = fx

        if normalize_output:
            output = layer_norm(output)

        return output
