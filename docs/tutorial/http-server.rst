Using ``bert-as-service`` to serve HTTP requests in JSON
========================================================

Besides calling ``bert-as-service`` from Python, one can also call it
via HTTP request in JSON. It is quite useful especially when low
transport layer is prohibited. Behind the scene, ``bert-as-service``
spawns a Flask server in a separate process and then reuse a
``BertClient`` instance as a proxy to communicate with the ventilator.

To enable this feature, we need to first install some Python
dependencies:

.. highlight:: bash
.. code:: bash

   pip install -U bert-serving-client flask flask-compress flask-cors flask-json

Then simply start the server with:

.. highlight:: bash
.. code:: bash

   bert-serving-start -model_dir=/YOUR_MODEL -http_port 8125

Your server is now listening HTTP and TCP requests at port ``8125``
simultaneously!

To send a HTTP request, first package payload in JSON as following:

.. highlight:: json
.. code:: json

   {
       "id": 123,
       "texts": ["hello world", "good day!"],
       "is_tokenized": false
   }

, where ``id`` is a unique identifier helping you to synchronize the
results; ``is_tokenized`` follows the meaning in ```BertClient`` API`_
and ``false`` by default.

Then simply call the server via HTTP POST request. You can use
javascript or whatever, here is an example using ``curl``:

.. highlight:: bash
.. code:: bash

   curl -X POST http://xx.xx.xx.xx:8125/encode \
     -H 'content-type: application/json' \
     -d '{"id": 123,"texts": ["hello world"], "is_tokenized": false}'

, which returns a JSON:

.. highlight:: json
.. code:: json

   {
       "id": 123,
       "results": [[768 float-list], [768 float-list]],
       "status": 200
   }


To get the server's status and client's status, you can send GET requests at
``/status/server`` and ``/status/client``, respectively.

Finally, one may also config CORS to restrict the public access of the
server by specifying ``-cors`` when starting ``bert-serving-start``. By
default ``-cors=*``, meaning the server is public accessible.

.. _``BertClient`` API: https://bert-as-service.readthedocs.io/en/latest/source/client.html#client.BertClient.encode_async