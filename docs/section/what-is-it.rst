What is it
==========

.. contents:: :local:

Preliminary
-----------

**BERT** is a NLP model `developed by Google <https://github.com/google-research/bert>`_ for pre-training language representations. It leverages an enormous amount of plain text data publicly available on the web and is trained in an unsupervised manner. Pre-training a BERT model is a fairly expensive yet one-time procedure for each language. Fortunately, Google released several pre-trained models where `you can download from here <https://github.com/google-research/bert#pre-trained-models>`_.

**Sentence Encoding/Embedding** is a upstream task required in many NLP applications, e.g. sentiment analysis, text classification. The goal is to represent a variable length sentence into a fixed length vector, e.g. hello world to [0.1, 0.3, 0.9]. Each element of the vector should "encode" some semantics of the original sentence.

**Finally,** ``bert-as-service`` uses BERT as a sentence encoder and hosts it as a service via ZeroMQ, allowing you to map sentences into fixed-length representations in just two lines of code.


Highlights
----------

- 🔭 State-of-the-art: build on pretrained 12/24-layer BERT models released by Google AI, which is considered as a milestone in the NLP community.
- 🐣 Easy-to-use: require only two lines of code to get sentence/token-level encodes.
- ⚡️ Fast: 900 sentences/s on a single Tesla M40 24GB. Low latency, optimized for speed. See benchmark.
- 🐙 Scalable: scale nicely and smoothly on multiple GPUs and multiple clients without worrying about concurrency. See benchmark.

More features: asynchronous encoding, multicasting, mix GPU & CPU workloads, graph optimization, tf.data friendly, customized tokenizer, pooling strategy and layer, XLA support etc.
