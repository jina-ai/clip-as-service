# Retrieval in CLIP-as-service
### Basics of retrieval
Retrieval is one of the most common use cases for embeddings. Usually retrieval contains two parts: encoding and indexing:

```{figure} images/retreival.png
:width: 80%
```

### Multi-modality retrieval in `CLIP-as-service`
`CLIP-as-service` offers us high-quality embeddings for [multi-modality data](https://docs.jina.ai/get-started/what-is-cross-modal-multi-modal/#what-is-cross-modal-multi-modal). It enables us to achieve cross-modality search like text-image retrieval or image-text retreival. 
And retrieval in `CLIP-as-service` support indexing a very large dataset (millions/billions) and querying within 50ms, depending on the machine.

In order to implement retrieval, we add an [`AnnLite`](https://github.com/jina-ai/annlite) indexer executor(based on [`HNSW`](https://arxiv.org/abs/1603.09320)) after the encoder executor in CLIP-as-service.


## Fast search in CLIP-as-service

Similar to `clip_server`, you can directly use the YAML config we have prepared for you by simple running:

```bash
python -m clip_server search_flow.yaml
```

The YAML config looks like this:
```yaml
jtype: Flow
version: '1'
with:
  port: 51000
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
        dim: 512
      metas:
        py_modules:
          - annlite.executor
```

| Parameter               | Description                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `dim`                  | Dimension of embeddings. Default is 512 since the output dimension of `CLIP` model is 512.|

Then indexing and searching are easy in `CLIP-as-service`:

```python
from clip_client import Client
from docarray import Document

client = Client('grpc://0.0.0.0:23456')

# index
client.index(
    [
        Document(text='she smiled, with pain'),  # text
        Document(uri='apple.png'),  # local image
        Document(
            uri='https://clip-as-service.jina.ai/_static/favicon.png'
        ),  # online image
    ]
)

# search
client.search(['smile'])
```

The results will look like this, the most relevant doc is "she smiled, with pain" with the cosine distance of 0.096. And the apple image has the cosine distance of 0.799.
```text
she smiled, with pain defaultdict(<class 'docarray.score.NamedScore'>, {'cosine': {'value': 0.09604912996292114}})
defaultdict(<class 'docarray.score.NamedScore'>, {'cosine': {'value': 0.7994112372398376}})
```

You don't need to call `client.encode()` explicitly since `client.index()` will handle this for you.



## How to lower memory footprint?
Sometimes the indexer will use a lot of memory because the HNSW indexer (which is used by `AnnLite`) is stored in memory. The efficient way to reduce memory footprint is dimension reduction. Retrieval in CLIP-as-service use [`Principal component analysis(PCA)`](https://en.wikipedia.org/wiki/Principal_component_analysis#:~:text=Principal%20component%20analysis%20(PCA)%20is,components%20and%20ignoring%20the%20rest.) to achieve this.

### Whether PCA is needed in my case?
It's hard to give an exactly number of how much memory should be used before you start indexing, but here are some facts that you can refer to:
- Memory usage is **linear** to the data size
- **1 million data (dim=512)** will approximately need **6G-7G memory**
- Actual memory usage will be 3-5 times as the [theoretical memory usage of HNSW](https://github.com/nmslib/hnswlib/issues/37)

So you can have an approximate estimation for memory usage based on your data size and dimension. And then compare it with the memory limit of your machine.

### Training PCA
In order to train a PCA model, you need to prepare training data.

The type of `train_data` is `numpy.ndarray` which are the embeddings you have prepared for training PCA. `train_data` can be obtained by simply using:

```python
results = client.encode(DocumentArray(...))  # your DocumentArray here
train_data = results.embeddings
```

After training data is prepared, you can start training PCA model use following script:
```python
from annlite.index import AnnLite
import numpy as np

index = AnnLite(dim=512, n_components=128)
index.train(train_data)
```

| Parameter               | Description                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `dim`                  | Dimension of embeddings. The output of the encoder as well as the input of PCA.|
| `n_components` | Output dimension of PCA.|

Once the training is done you will see the following outputs:
```text
2022-08-23 15:55:42.360 | INFO     | annlite.index:__init__:105 - Initialize Projector codec (n_components=128)
2022-08-23 15:55:42.375 | INFO     | annlite.index:train:191 - Start training Projector codec (n_components=128) with 1000 data...
2022-08-23 15:55:42.709 | INFO     | annlite.index:train:208 - The annlite is successfully trained!
2022-08-23 15:55:42.709 | INFO     | annlite.index:dump_model:543 - Save the parameters to workspace/parameters-dc287b278624be46d50b0d5cf7f9d59f
```

You will see a folder called `parameters-dc287b278624be46d50b0d5cf7f9d59f` under `workspace` which stores the PCA model.


```{tip}
There is no need to use the whole dataset to train PCA. But the number of training data should not be less than the original dimension of embeddings: 512 in this case.
```

### Load PCA model in server
In order to use PCA on the server side you need to add `n_components` inside the YAML config:
```yaml
jtype: Flow
version: '1'
with:
  port: 51000
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
        dim: 512  # input dimension of PCA
        n_components: 128  # output dimension of PCA
      metas:
        py_modules:
          - annlite.executor
```

After the service starts, `AnnLiteIndexer` will automatically load the PCA model we have trained before.

### Memory usage before and after PCA
Here is the comparison of memory usage before and after PCA when indexing 10 million data:

- X-axis: the number of data we have indexed.
- Y-axis: total memory usage.

```{figure} images/memory_usage_dim_512.png
:width: 80%
```
```{figure} images/memory_usage_dim_128.png
:width: 80%
```

Now the memory usage for indexing 10 million data decreases from 60GB+ to 30GB, saving more than 50% memory.

```{Note}
We use a single machine which has 90GB memory and 20 cores CPU.
```

```{Tip}
However, PCA will definitely lead to information losses since we remove some dimensions. And the more dimensions you remove, the more information losses will be. So the best practice will be estimate the memory usage first (if possible, see below) and choose the reasonable dimension after PCA.
```

## How to deal with a very large dataset?
For a very large dataset, for example, 100 million data or even 1 billion data, it's not possible to implement index operations on a single machine. **Sharding**, a type of partitioning that separates a large dataset into smaller, faster, more easily managed parts, is needed in this case.

You need to specify the `shards` and `polling` in the YAML config:
```yaml
jtype: Flow
version: '1'
with:
  port: 51000
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
        dim: 512  # input dimension of PCA
        n_components: 128  # output dimension of PCA
      metas:
        py_modules:
          - annlite.executor
    shards: 5
    polling: {'/index': 'ANY', '/search': 'ALL', '/update': 'ALL',
              '/delete': 'ALL', '/status': 'ALL'}
```

| Parameter               | Description                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `shards`                  | Number of shardings. |
| `polling` | Polling strategies for different endpoints.|

Then you can perform exactly the same operations as we do on a single machine.(`/encode`, `/index` and `/search`)

### Why different [polling strategies](https://docs.jina.ai/how-to/scale-out/?highlight=polling#different-polling-strategies) are needed for different endpoints?

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


## How to deploy it on the cloud?
Deployment can be easily achieved by using [`jcloud`](https://github.com/jina-ai/jcloud) or [`Amazon Kubernetes(EKS) Cluster`](https://aws.amazon.com/eks/). Taking `jcloud` as an example:

One can deploy a Jina Flow on `jcloud` by running the following command:
```bash
jc deploy search_flow.yml
```

The Flow is successfully deployed when you see:
```text
╭────────────── 🎉 Flow is available! ──────────────╮
│                                                   │
│   ID            418954ad0d                        │
│   Endpoint(s)   grpcs://418954ad0d.wolf.jina.ai   │
│                                                   │
╰───────────────────────────────────────────────────╯
```

One can send a request to Flow:
```python
from jina import Client, Document

c = Client(host='https://418954ad0d.wolf.jina.ai')
c.post('/index', Document(text='hello'))
```
