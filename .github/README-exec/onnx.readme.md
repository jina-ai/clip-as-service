# CLIPOnnxEncoder

**CLIPOnnxEncoder** is the executor implemented by [clip-as-service](https://github.com/jina-ai/clip-as-service). 
It serves OpenAI released [CLIP](https://github.com/openai/CLIP) models with ONNX runtime (🚀 **3x** speed up). 
The introduction of the CLIP model [can be found here](https://openai.com/blog/clip/).

- 🔀 **Automatic**: Auto-detect image and text documents depending on their content.
- ⚡ **Efficiency**: Non-blocking duplex streaming on requests and responses. 
- 📈 **Observability**: Monitoring the serving via Prometheus and Grafana.


## Model support

Open AI has released 9 models so far. `ViT-B/32` is used as default model in all runtimes. Please also note that **different model give different size of output dimensions**. This will affect your downstream applications. For example, switching the model from one to another make your embedding incomparable, which breaks the downstream applications. Here is a list of supported models of each runtime and its corresponding size:

| Model          | ONNX   | Output dimension | 
|----------------|-----| --- |
| RN50           | ✅   | 1024 | 
| RN101          | ✅   | 512 | 
| RN50x4         | ✅   | 640 |
| RN50x16        | ✅   | 768 |
| RN50x64        | ✅   | 1024 |
| ViT-B/32       | ✅   | 512 |
| ViT-B/16       | ✅   | 512 |
| ViT-L/14       | ✅   | 768 |
| ViT-L/14@336px | ✅   | 768 |

## Usage

### Use in Jina Flow 

- **via Docker image (recommended)**

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
    uses='jinahub+docker://CLIPOnnxEncoder',
)
```

- **via source code**

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
    uses='jinahub://CLIPOnnxEncoder',
)
```

You can set the following parameters via `with`:

| Parameter | Description                                                                                                                    |
|-----------|--------------------------------------------------------------------------------------------------------------------------------|
| `name`    | Model weights, default is `ViT-B/32`. Support all OpenAI released pretrained models.                                           |
| `num_worker_preprocess` | The number of CPU workers for image & text prerpocessing, default 4.                                                           | 
| `minibatch_size` | The size of a minibatch for CPU preprocessing and GPU encoding, default 64. Reduce the size of it if you encounter OOM on GPU. |
| `device`  | `cuda` or `cpu`. Default is `None` means auto-detect.                                                                          |

### Encoding

Encoding here means getting the fixed-length vector representation of a sentence or image.

```python
from jina import Flow
from docarray import Document, DocumentArray

da = DocumentArray(
    [
        Document(text='she smiled, with pain'),
        Document(uri='apple.png'),
        Document(uri='apple.png').load_uri_to_image_tensor(),
        Document(blob=open('apple.png', 'rb').read()),
        Document(uri='https://clip-as-service.jina.ai/_static/favicon.png'),
        Document(
            uri='data:image/gif;base64,R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7'
        ),
    ]
)

f = Flow().add(
    uses='jinahub+docker://CLIPOnnxEncoder',
)
with f:
    f.post(on='/', inputs=da)
```

we can get a summary of the results.

```text
╭──────────────────────────── Documents Summary ─────────────────────────────╮
│                                                                            │
│   Length                        6                                          │
│   Homogenous Documents          False                                      │
│   4 Documents have attributes   ('id', 'mime_type', 'uri', 'embedding')    │
│   1 Document has attributes     ('id', 'mime_type', 'text', 'embedding')   │
│   1 Document has attributes     ('id', 'embedding')                        │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── Attributes Summary ───────────────────────╮
│                                                                 │
│   Attribute   Data type      #Unique values   Has empty value   │
│  ─────────────────────────────────────────────────────────────  │
│   embedding   ('ndarray',)   6                False             │
│   id          ('str',)       6                False             │
│   mime_type   ('str',)       5                False             │
│   text        ('str',)       2                False             │
│   uri         ('str',)       4                False             │
│                                                                 │
╰─────────────────────────────────────────────────────────────────╯
```

To get the embedding of all Documents, simply call `da.embeddings`:

```text
[[-0.09136295  0.42720157 -0.05784469 ... -0.42873043  0.04472527
   0.4437953 ]
 [ 0.43152636  0.1563695  -0.09363698 ... -0.11514216  0.1865044
   0.15025651]
 [ 0.43152636  0.1563695  -0.09363698 ... -0.11514216  0.1865044
   0.15025651]
 [ 0.42862126  0.17757078  0.08584607 ...  0.23284511 -0.00929402
   0.10993651]
 [ 0.4706376  -0.01384148  0.3877237  ...  0.1995864  -0.22621225
  -0.4837676 ]]
```

👉 Access the embedding playground in **clip-as-service** [doc](https://clip-as-service.jina.ai/playground/embedding), type sentence or image URL and see **live embedding**!

### Ranking

One can also rank cross-modal matches via `/rank` endpoint. 
First construct a *cross-modal* Document where the root contains an image and `.matches` contain sentences to rerank. 

```python
from docarray import Document

d = Document(
    uri='rerank.png',
    matches=[
        Document(text=f'a photo of a {p}')
        for p in (
            'control room',
            'lecture room',
            'conference room',
            'podium indoor',
            'television studio',
        )
    ],
)
```

Then send the request via `/rank` endpoint:

```python
f = Flow().add(
    uses='jinahub+docker://CLIPOnnxEncoder',
)
with f:
    r = f.post(on='/rank', inputs=da)
    print(r['@m', ['text', 'scores__clip_score__value']])
```
Finally, in the return you can observe the matches are re-ranked according to `.scores['clip_score']`:

```bash
[['a photo of a television studio', 'a photo of a conference room', 'a photo of a lecture room', 'a photo of a control room', 'a photo of a podium indoor'], 
[0.9920725226402283, 0.006038925610482693, 0.0009973491542041302, 0.00078492151806131, 0.00010626466246321797]]
```

One can also construct `text-to-image` rerank as below:

```python
from docarray import Document

d = Document(
    text='a photo of conference room',
    matches=[
        Document(uri='https://picsum.photos/300'),
        Document(uri='https://picsum.photos/id/331/50'),
        Document(uri='https://clip-as-service.jina.ai/_static/favicon.png'),
    ],
)
```

👉 Access the ranking playground in **clip-as-service** [doc](https://clip-as-service.jina.ai/playground/reasoning/). Just input the reasoning texts as prompts, the server will rank the prompts and return sorted prompts with scores.