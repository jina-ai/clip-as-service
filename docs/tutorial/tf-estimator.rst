Training a text classifier using BERT features and ``tf.estimator`` API
=======================================================================

.. contents:: :local:


.. note:: The complete example can `be found example5.py`_, in which a simple MLP is built on BERT features for predicting the relevant articles according to the fact description in the law documents. The problem is a part of the `Chinese AI and Law Challenge Competition`_.

.. _be found example5.py: https://github.com/hanxiao/bert-as-service/blob/master/example/example5.py

Following the last example, we can easily extend it to a full classifier
using ``tf.estimator`` API. One only need minor change on the input
function as follows:

.. code:: python

   estimator = DNNClassifier(
       hidden_units=[512],
       feature_columns=[tf.feature_column.numeric_column('feature', shape=(768,))],
       n_classes=len(laws),
       config=run_config,
       label_vocabulary=laws_str,
       dropout=0.1)

   input_fn = lambda fp: (tf.data.TextLineDataset(fp)
                          .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
                          .batch(batch_size)
                          .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string]), num_parallel_calls=num_parallel_calls)
                          .map(lambda x, y: ({'feature': x}, y))
                          .prefetch(20))

   train_spec = TrainSpec(input_fn=lambda: input_fn(train_fp))
   eval_spec = EvalSpec(input_fn=lambda: input_fn(eval_fp), throttle_secs=0)
   train_and_evaluate(estimator, train_spec, eval_spec)


.. _Chinese AI and Law Challenge Competition: https://github.com/thunlp/CAIL/blob/master/README_en.md
