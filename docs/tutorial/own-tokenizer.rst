Using your own tokenizer
========================

.. contents:: :local:


Often you want to use your own tokenizer to segment sentences instead of
the default one from BERT. Simply call ``encode(is_tokenized=True)`` on
the client slide as follows:

.. highlight:: python
.. code:: python

   texts = ['hello world!', 'good day']

   # a naive whitespace tokenizer
   texts2 = [s.split() for s in texts]

   vecs = bc.encode(texts2, is_tokenized=True)

This gives ``[2, 25, 768]`` tensor where the first ``[1, 25, 768]``
corresponds to the token-level encoding of "hello world!". If you look
into its values, you will find that only the first four elements, i.e.
``[1, 0:3, 768]`` have values, all the others are zeros. This is due to
the fact that BERT considers "hello world!" as four tokens: ``[CLS]``
``hello`` ``world!`` ``[SEP]``, the rest are padding symbols and are
masked out before output.

.. note:: There is no need to start a separate server for handling tokenized/untokenized sentences. The server can tell and handle both cases automatically.


.. warning:: The pretrained BERT Chinese from Google is character-based, i.e. its vocabulary is made of single Chinese characters. Therefore it makes no sense if you use word-level segmentation algorithm to pre-process the data and feed to such model.