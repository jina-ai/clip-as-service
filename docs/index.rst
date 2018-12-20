.. bert-as-service documentation master file, created by
sphinx-quickstart on Wed Dec 19 20:32:51 2018.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

bert-as-service Documentation
=============================

``bert-as-service`` is a sentence encoding service for mapping a variable-length sentence to a fixed-length vector.

.. image:: ../.github/demo.gif

Preliminary
-----------

**BERT** is a NLP model `developed by Google <https://github.com/google-research/bert>`_ for pre-training language representations. It leverages an enormous amount of plain text data publicly available on the web and is trained in an unsupervised manner. Pre-training a BERT model is a fairly expensive yet one-time procedure for each language. Fortunately, Google released several pre-trained models where `you can download from here <https://github.com/google-research/bert#pre-trained-models>`_.

**Sentence Encoding/Embedding** is a upstream task required in many NLP applications, e.g. sentiment analysis, text classification. The goal is to represent a variable length sentence into a fixed length vector, e.g. hello world to [0.1, 0.3, 0.9]. Each element of the vector should "encode" some semantics of the original sentence.

**Finally,** ``bert-as-service`` uses BERT as a sentence encoder and hosts it as a service via ZeroMQ, allowing you to map sentences into fixed-length representations in just two lines of code.


.. toctree::
:maxdepth: 2
          source/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

