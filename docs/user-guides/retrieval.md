# Retrieval in CLIP-as-service
`CLIP-as-service` offers us high quality embeddings. Retrieval is one of the most common use cases for embeddings. Retrieval in CLIP-as-service can support indexing a very large dataset(millions/billions) and querying within 50ms.

In order to implement retrieval we add an extra indexer after enocoder in CLIP-as-service. We use [`AnnLite`](https://github.com/jina-ai/annlite) in this case. 


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


## How to lower memory footprint?
Sometimes the indexer will use a lot of memory because all embeddings and indexers are stored in memory. The efficient way to reduce memory footprint is dimension reduction. Retrieval in CLIP-as-service use [`Principal component analysis(PCA)`](https://en.wikipedia.org/wiki/Principal_component_analysis#:~:text=Principal%20component%20analysis%20(PCA)%20is,components%20and%20ignoring%20the%20rest.) to achieve this.


### Training PCA
In order to train a PCA model you need prepare train data. 
```python
from annlite.index import AnnLite
import numpy as np

index = AnnLite(dim=512, n_components=128, data_path='workspace')
index.train(train_data)
```

Here `train_data` is a `numpy.ndarray` which comes from embeddings you have already calculated. These embeddings can be obtained by simply calling `client.encode()`.

```{tip}
There is no need to use the whole dataset to train PCA. But the number of training data should not be less than the original dimension of embeddings: 512 in this case.
```

Once the training is done you will see following outputs:
```text
2022-08-23 15:55:42.360 | INFO     | annlite.index:__init__:105 - Initialize Projector codec (n_components=128)
2022-08-23 15:55:42.375 | INFO     | annlite.index:train:191 - Start training Projector codec (n_components=128) with 1000 data...
2022-08-23 15:55:42.709 | INFO     | annlite.index:train:208 - The annlite is successfully trained!
2022-08-23 15:55:42.709 | INFO     | annlite.index:dump_model:543 - Save the parameters to workspace/parameters-dc287b278624be46d50b0d5cf7f9d59f
```

You will see a folder called `parameters-dc287b278624be46d50b0d5cf7f9d59f` under `workspace`.

### Load PCA model in server
In order to enable PCA in server side you need to add `n_components` inside YAML config:
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

| Parameter               | Description                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `dim`                  | Dimension of embeddings. The output of the encoder as well as the input of PCA.|
| `n_components` | Output dimension of PCA.|


```{warning}
You must use the same `data_path` used in training PCA. 
```

### Memory usage before and after PCA
Here is the comparison of memory usage before and after PCA:

```{figure} images/memory_usage_dim_512.png

```
```{figure} images/memory_usage_dim_128.png

```

Now the memory usage for indexing 10 millions data decreases from 60GB+ to 30GB, saving more than 50% memory.

```{Warning}
However, PCA will definitely lead to information losses since we remove some dimensions. And more dimensions you remove, more information losses will be. So there will be a trade-off between efficiency and accuracy. 
```

### Whether PCA is needed in my case?
From our experiments, the memory usage is **linear** to the data size: **1 millions data with dimension of 512 will approximately need 8G-10G**. So you can have an approximately value for memory usage based on your data size.


## How to deal with very large dataset?
For a very large dataset, for example 100 millions data or even 1 billion data, it's not possible to implement index operations on a single machine. **Sharding**, a type of partitioning that separates large dataset into smaller, faster, more easily managed parts, is needed in this case.

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

Then you can perform exactly the same operations as we do in single machine.(`/encode`, `/index` and `/search`)

### Why different polling strategies are needed for different endpoints?

Differences between `ANY` and `ALL`:
- `ANY`: requests are send to one of executor.
- `ALL`: requests are send to all executors.

```{figure} images/polling_stratey.png
:width: 80%

```

Since one data point only needed to be indexed once, there will only be one indexer executor that will handle this data point. Thus, `ANY` is used for `/index`. On the contrary, search operations needed to be handle by all indexer executors since we don't know which executor store the perfectly matched result.

```{Warning}
Increase number of shardings will definitely alleviate the memory issue, but it will increase the latency since there will be more network connections between different shards.
```


## How to deploy it on cloud?