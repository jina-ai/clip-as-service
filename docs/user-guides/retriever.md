# Text & Image Search


CLIP Search is a search paradigm that uses the CLIP model to encode the text and image documents into a common vector space. 
The search results are then retrieved by computing the cosine similarity between the query and the indexed documents.
Technically, CLIP search can be designed as a two-stage process: *encoding* and *indexing*.

```{figure} images/retreival.png
:width: 80%
```

At the encoding stage, the text and image documents can be encoded into a common vector space by the CLIP model. 
It enables us to achieve cross-modal search, i.e., we can search for images given a text query, or search for text given an image query. 
At the indexing stage, we use the encoded vectors to build an index, which is a data structure that can be used to efficiently retrieve the most relevant documents.
Specifically, we use the [Annlite](https://github.com/jina-ai/annlite) indexer executor to build the index.

This chapter will walk you through the process of building a CLIP search system.


```{tip}
You will need to install server first in Python 3.7+: `pip install clip-server[search]>=0.7.0`.
```

## Start the server

To start the server, you can use the following command:

```bash
python -m clip_server search_flow.yml
```

The `search_flow.yml` is the yaml configuration file for the search flow. It defines a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) to implement the CLIP search system.
Below is an example of the Flow YAML file, we can put it into two subsections as below:

````{tab} CLIP model config

```{code-block} yaml
---
emphasize-lines: 9
---

jtype: Flow
version: '1'
with:
  port: 61000
executors:
  - name: encoder
    uses:
      jtype: CLIPEncoder
      with:
      metas:
        py_modules:
          - clip_server.executors.clip_torch
          
  - name: indexer
    uses:
      jtype: AnnLiteIndexer
      with:
        n_dim: 512
      workspace: './workspace'
      metas:
        py_modules:
          - annlite.executor
```

````

````{tab} Annlite indexer config

```{code-block} yaml
---
emphasize-lines: 17,18,19
---

jtype: Flow
version: '1'
with:
  port: 61000
executors:
  - name: encoder
    uses:
      jtype: CLIPEncoder
      with:
      metas:
        py_modules:
          - clip_server.executors.clip_torch
          
  - name: indexer
    uses:
      jtype: AnnLiteIndexer
      with:
        n_dim: 512
      workspace: './workspace'
      metas:
        py_modules:
          - annlite.executor
```

````

The first part defines the CLIP model config, which is explained [here](https://clip-as-service.jina.ai/user-guides/server/#clip-model-config).
And the second part defines the Annlite indexer config, you can set the following parameters:

| Parameter | Description                                                                                   |
|-----------|-----------------------------------------------------------------------------------------------|
| `n_dim`   | The dimension of the vector space. It should be the same as the dimension of the CLIP model.  |

And the `workspace` parameter is the path to the workspace directory, which is used to store the index files.

## Connect from client

```{tip}
You will need to install client first in Python 3.7+: `pip install clip-client>=0.7.0`.
```

To connect to the server, you can use the following code:

```python
from clip_client import Client
from docarray import Document

client = Client('grpc://0.0.0.0:61000')

# index
client.index(
    [
        Document(text='she smiled, with pain'),
        Document(uri='apple.png'),
        Document(uri='https://clip-as-service.jina.ai/_static/favicon.png'),
    ]
)

# search
result = client.search(['smile'])
```

The results will look like this, the most relevant doc is "she smiled, with pain" with the cosine distance of 0.096. And the apple image has the cosine distance of 0.799.
```text
she smiled, with pain defaultdict(<class 'docarray.score.NamedScore'>, {'cosine': {'value': 0.09604912996292114}})
defaultdict(<class 'docarray.score.NamedScore'>, {'cosine': {'value': 0.7994112372398376}})
```

You don't need to call `client.encode()` explicitly since `client.index()` will handle this for you.

## Support large-scale dataset

When we want to index a large number of documents, for example, 100 million data or even 1 billion data, 
it's not possible to implement index operations on a single machine. **Sharding**, 
a type of partitioning that separates a large dataset into smaller, faster, more easily managed parts, is needed in this case.

You need to specify the `shards` and `polling` in the YAML config:

```yaml
jtype: Flow
version: '1'
with:
  port: 61000
executors:
  - name: encoder
    uses:
      jtype: CLIPEncoder
      metas:
        py_modules:
          - clip_server.executors.clip_torch
          
  - name: indexer
    uses:
      jtype: AnnLiteIndexer
      with:
        n_dim: 512
      workspace: './workspace'
      metas:
        py_modules:
          - annlite.executor
    shards: 5
    polling: {'/index': 'ANY', '/search': 'ALL', '/update': 'ALL',
              '/delete': 'ALL', '/status': 'ALL'}
```

| Parameter   | Description                                 |
|-------------|---------------------------------------------|
| `shards`    | Number of shardings.                        |
| `polling`   | Polling strategies for different endpoints. |

Then you can perform exactly the same operations as we do on a single machine.(`/encode`, `/index` and `/search`)

**Why different [polling strategies](https://docs.jina.ai/how-to/scale-out/?highlight=polling#different-polling-strategies) are needed for different endpoints?**

Differences between `ANY` and `ALL`:
- `ANY`: requests are sent to one of the executors.
- `ALL`: requests are sent to all executors.

```{figure} images/polling_stratey.png
:width: 80%

```

Since one data point only needs to be indexed once, there will only be one indexer executor that will handle this data point. Thus, `ANY` is used for `/index`. On the contrary, we use `ALL` in for `/search` since we don't know which executor stores the perfectly matched result, so the search request should be handled by all indexer executors. (The same reason for using `ALL` in `/update`, `/delete`, `/status`)

```{Warning}
Increasing the number of shardings will definitely alleviate the memory issue, but it will increase the latency since there will be more network connections between different shards.
```
