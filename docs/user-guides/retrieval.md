# Retrieval in CLIP-as-service
`CLIP-as-service` offers us high quality embeddings. Retrieval is one of the most common use cases for embeddings. Retrieval in CLIP-as-service can support indexing a very large dataset(millions/billions) and querying within several million seconds.

In order to implement retrieval we add an extra indexer after enocoder in CLIP-as-service. We use ['AnnLite'](https://github.com/jina-ai/annlite) in this case. 


## Fast search in CLIP-as-service
Index and search are easy in `CLIP-as-service`:

```python
from clip_client import Client

client = Client('grpc://0.0.0.0:23456')

# index
r = client.index(
    [
        'she smiled, with pain',  # text
        'https://clip-as-service.jina.ai/_static/favicon.png',  # image
    ]
)

# search
client.search(['smile'])
```

The results will be like:

```text
╭───────────────────────────── Documents Summary ─────────────────────────────╮
│                                                                             │
│   Length                 1                                                  │
│   Homogenous Documents   True                                               │
│   Common Attributes      ('id', 'mime_type', 'text', 'tags', 'embedding')   │
│   Multimodal dataclass   False                                              │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── Attributes Summary ───────────────────────╮
│                                                                 │
│   Attribute   Data type      #Unique values   Has empty value   │
│  ─────────────────────────────────────────────────────────────  │
│   embedding   ('ndarray',)   1                False             │
│   id          ('str',)       1                False             │
│   mime_type   ('str',)       1                False             │
│   tags        ('dict',)      1                False             │
│   text        ('str',)       1                False             │
│                                                                 │
╰─────────────────────────────────────────────────────────────────╯
```

You don't need to call `client.encode()` explicitly since `client.index()` will handle this for you.


Similar to `CLIP-as-service`, you need to prepare a YAML config:
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
  	  	data_path: workspace
  	  metas:
  	  	py_modules:
  	  	  - annlite.executor
```

| Parameter               | Description                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `dim`                  | Dimension of embeddings. Default is 512 since the output dimension of `CLIP` model is 512.|
| `data_path` | Path of indexer files. Default is `workspace`|


## How to lower memory footprint?
Sometimes the indexer will use a lot of memory because all embeddings and indexers are stored in memory. The efficient way to reduce memory footprint is dimension reduction. Retrieval in CLIP-as-service use [`Principal component analysis(PCA)`](https://en.wikipedia.org/wiki/Principal_component_analysis#:~:text=Principal%20component%20analysis%20(PCA)%20is,components%20and%20ignoring%20the%20rest.) to achieve this.

In order to enable PCA you only need to add `n_components` inside YAML config:
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
  	  	data_path: workspace
  	  	n_components: 128  # output dimension of PCA
  	  metas:
  	  	py_modules:
  	  	  - annlite.executor
```

| Parameter               | Description                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `dim`                  | Dimension of embeddings. The output of the encoder as well as the input of PCA.|
| `n_components` | Output dimension of PCA.|


Then you need to train a PCA model:


Here is the comparison of memory usage before and after PCA:

```{figure} images/memory_usage_dim_512.png
:width: 80%

```
```{figure} images/memory_usage_dim_128.png
:width: 80%

```

Now the memory usage for indexing 10 millions data decreases from 60GB+ to 30GB, saving more than 50% memory.

However, PCA will definitely lead to information losses since we remove some dimensions. And more dimensions you remove, more information losses will be. So there will be a trade-off between efficiency and accuracy. 

### How do I know whether PCA is needed in my case?
From our experiments, the memory usage is **linear** to the data size: **1 millions data with dimension of 512 will approximately need 8G-10G**. So you can have an approximately value for memory usage based on your data size.


## How to deal with very large dataset?
For a very large dataset, for example 100M or even 1B, it's not possible to implement index operation on a single machine. **Sharding**, a type of partitioning that separates large dataset into smaller, faster, more easily managed parts, is needed in this case.

You need to speicify the `shards` and `polling` in YAML config:
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
  	  	data_path: workspace
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
| `polling` | Polling strategy for different endpoints.|


### Why different polling strategies are needed for different endpoints?

Differences between `ANY` and `ALL`:
- `ANY`: requests are send to one of executor.
- `ALL`: requests are send to all executors.

```{figure} images/polling_stratey.png
:width: 80%

```

Since one data point only needed to be indexed once, there will only be one indexer executor that will handle this data point. Thus, `ANY` is used for `/index`. On the contrary, search operations needed to be handle by all indexer executors since we don't know which executor store the perfectly matched result.


## How to deploy it on cloud?