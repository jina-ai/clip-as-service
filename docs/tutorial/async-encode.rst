Asynchronous encoding
=====================

.. contents:: :local:


.. note:: The complete example can `be found example2.py`_.

.. _be found example2.py: https://github.com/hanxiao/bert-as-service/blob/master/example/example2.py

``BertClient.encode()`` offers a nice synchronous way to get sentence
encodes. However, sometimes we want to do it in an asynchronous manner
by feeding all textual data to the server first, fetching the encoded
results later. This can be easily done by:

.. highlight:: python
.. code:: python

   # an endless data stream, generating data in an extremely fast speed
   def text_gen():
       while True:
           yield lst_str  # yield a batch of text lines

   bc = BertClient()

   # get encoded vectors
   for j in bc.encode_async(text_gen(), max_num_batch=10):
       print('received %d x %d' % (j.shape[0], j.shape[1]))