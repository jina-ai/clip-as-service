# CLIPOnnxEncoder

`CLIPOnnxEncoder` serve OpenAI released [CLIP](https://github.com/openai/CLIP) models with ONNX runtime. 
It embeds documents using either the text or the visual part of CLIP, depending on their content.

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
    uses='jinahub+docker://CLIPOnnxEncoder',
)

with f:
    f.post('/', inputs=[Document(text='hello world') for _ in range(3)])
    f.post('/', inputs=[Document(uri='https://picsum.photos/200') for _ in range(3)])
```

#### via source code

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
    uses='jinahub://CLIPOnnxEncoder',
)

with f:
    f.post('/', inputs=[Document(text='hello world') for _ in range(3)])
    f.post('/', inputs=[Document(uri='https://picsum.photos/200') for _ in range(3)])
```



## CRUD Operations

You can perform CRUD operations (create, read, update and delete) using the respective endpoints:

- `/index`: Add new data to `Weaviate`. 
- `/search`: Query the `Weaviate` index (created in `/index`) with your Documents.
- `/update`: Update Documents in `Weaviate`.
- `/delete`: Delete Documents in `Weaviate`.


## Vector Search

The following example shows how to perform vector search using`f.post(on='/search', inputs=[Document(embedding=np.array([1,1]))])`.


```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
    uses='jinahub://WeaviateIndexer',
    uses_with={'n_dim': 2},
)

with f:
    f.post(
        on='/index',
        inputs=[
            Document(id='a', embedding=np.array([1, 3])),
            Document(id='b', embedding=np.array([1, 1])),
        ],
    )

    docs = f.post(
        on='/search',
        inputs=[Document(embedding=np.array([1, 1]))],
    )

# will print "The ID of the best match of [1,1] is: b"
print('The ID of the best match of [1,1] is: ', docs[0].matches[0].id)
```


### Using filtering
To do filtering with the WeaviateIndexer you should first define columns and precise the dimension of your embedding space.
For instance :


```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://WeaviateIndexer',
    uses_with={
        'name': 'Test',
        'n_dim': 3,
        'columns': [('price', 'float')],
    },
)
```

Then you can pass a filter as a parameters when searching for document:
```python
from docarray import Document, DocumentArray
import numpy as np

docs = DocumentArray(
    [
        Document(id=f'r{i}', embedding=np.random.rand(3), tags={'price': i})
        for i in range(50)
    ]
)


filter_ = {'path': ['price'], 'operator': 'LessThanEqual', 'valueInt': 30}

with f:
    f.index(docs)
    doc_query = DocumentArray([Document(embedding=np.random.rand(3))])
    f.search(doc_query, parameters={'filter': filter_})
```

For more information please refer to the docarray [documentation](https://docarray.jina.ai/advanced/document-store/weaviate/#vector-search-with-filter)