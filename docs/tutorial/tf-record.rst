Saving and loading with TFRecord data
=====================================

.. contents:: :local:


.. note:: The complete example can `be found example6.py`_.

.. _be found example6.py: https://github.com/hanxiao/bert-as-service/blob/master/example/example6.py


The TFRecord file format is a simple record-oriented binary format that
many TensorFlow applications use for training data. You can also
pre-encode all your sequences and store their encodings to a TFRecord
file, then later load it to build a ``tf.Dataset``. For example, to
write encoding into a TFRecord file:

.. highlight:: python
.. code:: python

   bc = BertClient()
   list_vec = bc.encode(lst_str)
   list_label = [0 for _ in lst_str]  # a dummy list of all-zero labels

   # write to tfrecord
   with tf.python_io.TFRecordWriter('tmp.tfrecord') as writer:
       def create_float_feature(values):
           return tf.train.Feature(float_list=tf.train.FloatList(value=values))

       def create_int_feature(values):
           return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

       for (vec, label) in zip(list_vec, list_label):
           features = {'features': create_float_feature(vec), 'labels': create_int_feature([label])}
           tf_example = tf.train.Example(features=tf.train.Features(feature=features))
           writer.write(tf_example.SerializeToString())

Now we can load from it and build a ``tf.Dataset``:

.. highlight:: python
.. code:: python

   def _decode_record(record):
       """Decodes a record to a TensorFlow example."""
       return tf.parse_single_example(record, {
           'features': tf.FixedLenFeature([768], tf.float32),
           'labels': tf.FixedLenFeature([], tf.int64),
       })

   ds = (tf.data.TFRecordDataset('tmp.tfrecord').repeat().shuffle(buffer_size=100).apply(
       tf.contrib.data.map_and_batch(lambda record: _decode_record(record), batch_size=64))
         .make_one_shot_iterator().get_next())

To save word/token-level embedding to TFRecord, one needs to first
flatten ``[max_seq_len, num_hidden]`` tensor into an 1D array as
follows:

.. highlight:: python
.. code:: python

   def create_float_feature(values):
       return tf.train.Feature(float_list=tf.train.FloatList(value=values.reshape(-1)))

And later reconstruct the shape when loading it:

.. highlight:: python
.. code:: python

   name_to_features = {
       "feature": tf.FixedLenFeature([max_seq_length * num_hidden], tf.float32),
       "label_ids": tf.FixedLenFeature([], tf.int64),
   }

   def _decode_record(record, name_to_features):
       """Decodes a record to a TensorFlow example."""
       example = tf.parse_single_example(record, name_to_features)
       example['feature'] = tf.reshape(example['feature'], [max_seq_length, -1])
       return example

Be careful, this will generate a huge TFRecord file.