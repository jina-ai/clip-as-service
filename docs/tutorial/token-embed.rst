Getting ELMo-like contextual word embedding
===========================================

.. contents:: :local:


Start the server with ``pooling_strategy`` set to NONE.

.. highlight:: bash
.. code:: bash

   bert-serving-start -pooling_strategy NONE -model_dir /tmp/english_L-12_H-768_A-12/

To get the word embedding corresponds to every token, you can simply use
slice index as follows:

.. highlight:: python
.. code:: python

   # max_seq_len = 25
   # pooling_strategy = NONE

   bc = BertClient()
   vec = bc.encode(['hey you', 'whats up?'])

   vec  # [2, 25, 768]
   vec[0]  # [1, 25, 768], sentence embeddings for `hey you`
   vec[0][0]  # [1, 1, 768], word embedding for `[CLS]`
   vec[0][1]  # [1, 1, 768], word embedding for `hey`
   vec[0][2]  # [1, 1, 768], word embedding for `you`
   vec[0][3]  # [1, 1, 768], word embedding for `[SEP]`
   vec[0][4]  # [1, 1, 768], word embedding for padding symbol
   vec[0][25]  # error, out of index!

Note that no matter how long your original sequence is, the service will
always return a ``[max_seq_len, 768]`` matrix for every sequence. When
using slice index to get the word embedding, beware of the special
tokens padded to the sequence, i.e. ``[CLS]``, ``[SEP]``, ``0_PAD``.