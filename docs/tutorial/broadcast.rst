Broadcasting to multiple clients
================================

.. contents:: :local:


.. note:: The complete example can `be found example3.py`_.

.. _be found example3.py: https://github.com/hanxiao/bert-as-service/blob/master/example/example3.py

The encoded result is routed to the client according to its identity. If
you have multiple clients with same identity, then they all receive the
results! You can use this *multicast* feature to do some cool things,
e.g. training multiple different models (some using ``scikit-learn``
some using ``tensorflow``) in multiple separated processes while only
call ``BertServer`` once. In the example below, ``bc`` and its two
clones will all receive encoded vector.

.. highlight:: python
.. code:: python

   # clone a client by reusing the identity
   def client_clone(id, idx):
       bc = BertClient(identity=id)
       for j in bc.listen():
           print('clone-client-%d: received %d x %d' % (idx, j.shape[0], j.shape[1]))

   bc = BertClient()
   # start two cloned clients sharing the same identity as bc
   for j in range(2):
       threading.Thread(target=client_clone, args=(bc.identity, j)).start()

   for _ in range(3):
       bc.encode(lst_str)