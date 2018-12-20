Serving a fine-tuned BERT model
===============================

.. contents:: :local:


Pretrained BERT models often show quite "okayish" performance on many
tasks. However, to release the true power of BERT a fine-tuning on the
downstream task (or on domain-specific data) is necessary. In this
example, I will show you how to serve a fine-tuned BERT model.

We follow the instruction in `"Sentence (and sentence-pair)
classification tasks"`_ and use ``run_classifier.py`` to fine tune
``uncased_L-12_H-768_A-12`` model on MRPC task. The fine-tuned model is
stored at ``/tmp/mrpc_output/``, which can be changed by specifying
``--output_dir`` of ``run_classifier.py``.

If you look into ``/tmp/mrpc_output/``, it contains something like:

.. highlight:: bash
.. code:: bash

   checkpoint                                        128
   eval                                              4.0K
   eval_results.txt                                  86
   eval.tf_record                                    219K
   events.out.tfevents.1545202214.TENCENT64.site     6.1M
   events.out.tfevents.1545203242.TENCENT64.site     14M
   graph.pbtxt                                       9.0M
   model.ckpt-0.data-00000-of-00001                  1.3G
   model.ckpt-0.index                                23K
   model.ckpt-0.meta                                 3.9M
   model.ckpt-343.data-00000-of-00001                1.3G
   model.ckpt-343.index                              23K
   model.ckpt-343.meta                               3.9M
   train.tf_record                                   2.0M

Don't be afraid of those mysterious files, as the only important one to
us is ``model.ckpt-343.data-00000-of-00001`` (looks like my training
stops at the 343 step. One may get
``model.ckpt-123.data-00000-of-00001`` or
``model.ckpt-9876.data-00000-of-00001`` depending on the total training
steps). Now we have collected all three pieces of information that are
needed for serving this fine-tuned model:

-  The pretrained model is downloaded to
   ``/path/to/bert/uncased_L-12_H-768_A-12``
-  Our fine-tuned model is stored at ``/tmp/mrpc_output/``;
-  Our fine-tuned model checkpoint is named as ``model.ckpt-343``
   something something.

Now start a BertServer by putting three pieces together:

.. highlight:: bash
.. code:: bash

   bert-serving-start -model_dir=/pretrained/uncased_L-12_H-768_A-12 -tuned_model_dir=/tmp/mrpc_output/ -ckpt_name=model.ckpt-343

After the server started, you should find this line in the log:

.. highlight:: bash
.. code:: text

   I:GRAPHOPT:[gra:opt: 50]:checkpoint (override by fine-tuned model): /tmp/mrpc_output/model.ckpt-343

Which means the BERT parameters is overrode and successfully loaded from
our fine-tuned ``/tmp/mrpc_output/model.ckpt-343``. Done!

In short, find your fine-tuned model path and checkpoint name, then feed
them to ``-tuned_model_dir`` and ``-ckpt_name``, respectively.

.. _"Sentence (and sentence-pair) classification tasks": https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks