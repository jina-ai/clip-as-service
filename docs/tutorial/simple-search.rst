Building a QA semantic search engine in 3 minutes
=================================================

.. contents:: :local:


.. note:: The complete example can `be found example8.py`_.

.. _be found example8.py: https://github.com/hanxiao/bert-as-service/blob/master/example/example8.py

As the first example, we will implement a simple QA search engine using
``bert-as-service`` in just three minutes. No kidding! The goal is to
find similar questions to user's input and return the corresponding
answer. To start, we need a list of question-answer pairs. Fortunately,
this README file already contains `a list of FAQ`_, so I will just use
that to make this example perfectly self-contained. Let's first load all
questions and show some statistics.

.. highlight:: python
.. code:: python

   prefix_q = '##### **Q:** '
   with open('README.md') as fp:
       questions = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
       print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))

This gives ``33 questions loaded, avg. len of 9``. So looks like we have
enough questions. Now start a BertServer with
``uncased_L-12_H-768_A-12`` pretrained BERT model:

.. highlight:: bash
.. code:: bash

   bert-serving-start -num_worker=1 -model_dir=/data/cips/data/lab/data/model/uncased_L-12_H-768_A-12

Next, we need to encode our questions into vectors:

.. highlight:: python
.. code:: python

   bc = BertClient(port=4000, port_out=4001)
   doc_vecs = bc.encode(questions)

Finally, we are ready to receive new query and perform a simple "fuzzy"
search against the existing questions. To do that, every time a new
query is coming, we encode it as a vector and compute its dot product
with ``doc_vecs``; sort the result descendingly; and return the top-k
similar questions as follows:

.. highlight:: python
.. code:: python

   while True:
       query = input('your question: ')
       query_vec = bc.encode([query])[0]
       # compute normalized dot product as score
       score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
       topk_idx = np.argsort(score)[::-1][:topk]
       for idx in topk_idx:
           print('> %s\t%s' % (score[idx], questions[idx]))

That's it! Now run the code and type your query, see how this search
engine handles fuzzy match:

.. image:: ../../.github/qasearch-demo.gif

.. _a list of FAQ: #speech_balloon-faq

