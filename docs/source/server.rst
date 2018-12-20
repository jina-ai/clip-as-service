Using BertServer
================

.. contents:: :local:

Installation
------------

The best way to install the server is via pip. Note that the server can be installed separately from BertClient or even on a different machine:

.. highlight:: bash
.. code-block:: bash

    pip install bert-serving-server

.. warning:: The server MUST be running on **Python >= 3.5** with **Tensorflow >= 1.10** (*one-point-ten*). Again, the server does not support Python 2!

Command Line Interface
----------------------

Once installed, you can use the command line interface to start a bert server:

.. highlight:: bash
.. code-block:: bash

    bert-serving-server -model_dir /uncased_bert_model -num_worker 4

Server-side API
---------------

Server-side is a CLI ``bert-serving-start``, you can get the latest
usage via:

.. code:: bash

   bert-serving-start --help

======================= ===== ==================== ========================================================================================================================================================================================================================================================================================================================================
Argument                Type  Default              Description
======================= ===== ==================== ========================================================================================================================================================================================================================================================================================================================================
``model_dir``           str   *Required*           folder path of the pre-trained BERT model.
``tuned_model_dir``     str   (Optional)           folder path of a fine-tuned BERT model.
``ckpt_name``           str   ``bert_model.ckpt``  filename of the checkpoint file.
``config_name``         str   ``bert_config.json`` filename of the JSON config file for BERT model.
``max_seq_len``         int   ``25``               maximum length of sequence, longer sequence will be trimmed on the right side.
``num_worker``          int   ``1``                number of (GPU/CPU) worker runs BERT model, each works in a separate process.
``max_batch_size``      int   ``256``              maximum number of sequences handled by each worker, larger batch will be partitioned into small batches.
``priority_batch_size`` int   ``16``               batch smaller than this size will be labeled as high priority, and jumps forward in the job queue to get result faster
``port``                int   ``5555``             port for pushing data from client to server
``port_out``            int   ``5556``             port for publishing results from server to client
``pooling_strategy``    str   ``REDUCE_MEAN``      the pooling strategy for generating encoding vectors, valid values are ``NONE``, ``REDUCE_MEAN``, ``REDUCE_MAX``, ``REDUCE_MEAN_MAX``, ``CLS_TOKEN``, ``FIRST_TOKEN``, ``SEP_TOKEN``, ``LAST_TOKEN``. Explanation of these strategies `can be found here`_. To get encoding for each token in the sequence, please set this to ``NONE``.
``pooling_layer``       list  ``[-2]``             the encoding layer that pooling operates on, where ``-1`` means the last layer, ``-2`` means the second-to-last, ``[-1, -2]`` means concatenating the result of last two layers, etc.
``gpu_memory_fraction`` float ``0.5``              the fraction of the overall amount of memory that each GPU should be allocated per worker
``cpu``                 bool  ``False``            run on CPU instead of GPU
``xla``                 bool  ``False``            enable `XLA compiler`_ for graph optimization (*experimental!*)
``device_map``          list  ``[]``               specify the list of GPU device ids that will be used (id starts from 0)
======================= ===== ==================== ========================================================================================================================================================================================================================================================================================================================================

.. _can be found here: #q-what-are-the-available-pooling-strategies
.. _XLA compiler: https://www.tensorflow.org/xla/jit
