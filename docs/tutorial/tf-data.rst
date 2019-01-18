Using BertClient with ``tf.data`` API
=====================================

.. contents:: :local:


.. note:: The complete example can `be found example4.py`_. There is also `an example in Keras`_.

.. _be found example4.py: https://github.com/hanxiao/bert-as-service/blob/master/example/example4.py
.. _an example in Keras: https://github.com/hanxiao/bert-as-service/issues/29#issuecomment-442362241


The ``tf.data`` `API`_ enables you to build complex input pipelines from
simple, reusable pieces. One can also use ``BertClient`` to encode
sentences on-the-fly and use the vectors in a downstream model. Here is
an example:

.. highlight:: python
.. code:: python

   batch_size = 256
   num_parallel_calls = 4

   # start a thead-safe client to support num_parallel_calls in tf.data API
   bc = ConcurrentBertClient(num_parallel_calls)


   def get_encodes(x):
      # x is `batch_size` of lines, each of which is a json object
      samples = [json.loads(l) for l in x]
      text = [s['raw_text'] for s in samples]  # List[List[str]]
      labels = [s['label'] for s in samples]  # List[str]
      features = bc.encode(text)
      return features, labels


   ds = (tf.data.TextLineDataset(train_fp).batch(batch_size)
           .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string]),  num_parallel_calls=num_parallel_calls)
           .map(lambda x, y: {'feature': x, 'label': y})
           .make_one_shot_iterator().get_next())

The trick here is to start a pool of ``BertClient`` and reuse them one
by one. In this way, we can fully harness the power of
``num_parallel_calls`` in ``Dataset.map()`` API.

.. _API: https://www.tensorflow.org/guide/datasets
