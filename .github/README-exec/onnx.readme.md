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

