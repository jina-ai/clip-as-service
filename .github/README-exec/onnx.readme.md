# CLIPOnnxEncoder

**CLIPOnnxEncoder** is the executor implemented in [CLIP-as-service](https://github.com/jina-ai/clip-as-service). 
The various `CLIP` models implemented in the [OpenAI](https://github.com/openai/CLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip) are supported with ONNX runtime (🚀 **3x** speed up). 
The introduction of the CLIP model [can be found here](https://openai.com/blog/clip/).

- 🔀 **Automatic**: Auto-detect image and text documents depending on their content.
- ⚡ **Efficiency**: Faster CLIP model inference on CPU and GPU via ONNX runtime. 
- 📈 **Observability**: Monitoring the serving via Prometheus and Grafana (see [Usage Guide](https://docs.jina.ai/how-to/monitoring/#deploying-locally)).


## Model support

 `ViT-B-32::openai` is used as the default model. To use specific pretrained models provided by `open_clip`, please use `::` to separate model name and pretrained weight name, e.g. `ViT-B-32::laion2b_e16`. Please also note that **different models give different sizes of output dimensions**.

| Model                                 | ONNX | Output dimension | 
|---------------------------------------|------|------------------|
| RN50                                  | ✅    | 1024             | 
| RN101                                 | ✅    | 512              | 
| RN50x4                                | ✅    | 640              |
| RN50x16                               | ✅    | 768              |
| RN50x64                               | ✅    | 1024             |
| ViT-B-32                              | ✅    | 512              |
| ViT-B-16                              | ✅    | 512              |
| ViT-B-16-plus-240                     | ✅    | 640              |
| ViT-L-14                              | ✅    | 768              |
| ViT-L-14-336                          | ✅    | 768              |
| ViT-H-14                              | ✅    | 1024             |
| ViT-g-14                              | ✅    | 1024             |
| M-CLIP/XLM_Roberta-Large-Vit-B-32     | ✅    | 512              |
| M-CLIP/XLM-Roberta-Large-Vit-L-14     | ✅    | 768              |
| M-CLIP/XLM-Roberta-Large-Vit-B-16Plus | ✅    | 640              |
| M-CLIP/LABSE-Vit-L-14                 | ✅    | 768              |

✅ = First class support 

Full list of open_clip models and weights can be found [here](https://github.com/mlfoundations/open_clip#pretrained-model-interface).

```{note}
For model definition with `-quickgelu` postfix, please use non `-quickgelu` model name.
```

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

| Parameter | Description                                                                                                                   |
|-----------|-------------------------------------------------------------------------------------------------------------------------------|
| `name`    | Model weights, default is `ViT-B/32`. Support all OpenAI released pretrained models.                                          |
| `num_worker_preprocess` | The number of CPU workers for image & text prerpocessing, default 4.                                                          | 
| `minibatch_size` | The size of a minibatch for CPU preprocessing and GPU encoding, default 16. Reduce the size of it if you encounter OOM on GPU. |
| `device`  | `cuda` or `cpu`. Default is `None` means auto-detect.                                                                         |

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
    da.summary()
```

From the output, you will see all the text and image docs have `embedding` attached.

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

👉 Access the embedding playground in **CLIP-as-service** [doc](https://clip-as-service.jina.ai/playground/embedding), type sentence or image URL and see **live embedding**!

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
    r = f.post(on='/rank', inputs=[d])
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

👉 Access the ranking playground in **CLIP-as-service** [doc](https://clip-as-service.jina.ai/playground/reasoning/). Just input the reasoning texts as prompts, the server will rank the prompts and return sorted prompts with scores.