# Server API

CLIP-as-service is designed in a client-server architecture. A server is a long-running program that receives raw sentences and images from clients, and returns CLIP embeddings to the client. Additionally, `clip_server` is optimized for speed, low memory footprint and scalability.
- Horizontal scaling: adding more replicas easily with one argument. 
- Vertical scaling: using PyTorch JIT or ONNX runtime to speedup single GPU inference.
- Supporting gRPC, HTTP, Websocket protocols with their TLS counterparts, w/o compressions.

This chapter introduces the API of the client. 

```{tip}
You will need to install client first in Python 3.7+: `pip install clip-server`.
```

## Start server

Unlike the client, server only has a CLI entrypoint. To start a server, run the following in the terminal:

```bash
python -m clip_server
```

Note that it is underscore `_` not the dash `-`.

(server-address)=
First time running will download the pretrained model (Pytorch `ViT-B/32` by default), load the model, and finally you will get the address information of the server. This information will {ref}`then be used in clients<construct-client>`.

```{figure} images/server-start.gif
:width: 70%

```

To use ONNX runtime for CLIP, you can run:

```bash
pip install "clip_server[onnx]"

python -m clip_server onnx_flow.yml
```

One may wonder where is this `onnx_flow.yml` come from. Must be a typo? Believe me, just run it. It should work. I will explain this YAML file in the next section. 

The procedure and UI of ONNX runtime would look the same as Pytorch runtime.



## YAML config

You may notice that there is a YAML file in our last ONNX example. All configurations are stored in this file. In fact, `python -m clip_server` does **not support** any other argument besides a YAML file. So it is the only source of the truth of your configs. 

And to answer your doubt, `clip_server` has two built-in YAML configs as a part of the package resources: one for PyTorch backend, one for ONNX backend. When you do `python -m clip_server` it loads the Pytorch config, and when you do `python -m clip_server onnx-flow.yml` it loads the ONNX config.

Let's look at these two built-in YAML configs:

````{tab} torch-flow.yml

```yaml
jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      metas:
        py_modules:
          - executors/clip_torch.py
```
````

````{tab} onnx-flow.yml

```yaml
jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_o
    uses:
      jtype: CLIPEncoder
      metas:
        py_modules:
          - executors/clip_onnx.py
```
````

Basically, each YAML file defines a [Jina Flow](https://docs.jina.ai/fundamentals/flow/). The complete Jina Flow YAML syntax [can be found here](https://docs.jina.ai/fundamentals/flow/flow-yaml/#configure-flow-meta-information). General parameters of the Flow and Executor can be used here as well. But now we only highlight the most important parameters.

Looking at the YAML file again, we can put it into three subsections as below:



````{tab} CLIP model config

```{code-block} yaml
---
emphasize-lines: 9
---

jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      with:
      metas:
        py_modules:
          - executors/clip_torch.py
```

````

````{tab} Executor config

```{code-block} yaml
---
emphasize-lines: 6
---

jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      with: 
      metas:
        py_modules:
          - executors/clip_torch.py
```

````

````{tab} Flow config

```{code-block} yaml
---
emphasize-lines: 3,4
---

jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      with: 
      metas:
        py_modules:
          - executors/clip_torch.py
```

````

### CLIP model config

For PyTorch backend, you can set the following parameters via `with`:

| Parameter | Description                                                                       |
|-----------|-----------------------------------------------------------------------------------|
| `name`    | Model weights, default is `ViT-B/32`. Support all OpenAI released pretrained models |
| `device`  | `cuda` or `cpu`. Default is `None` means auto-detect                              |
| `jit` | If to enable Torchscript JIT, default is `False`| 


For ONNX backend, you can set the following parameters:

| Parameter | Description                                                                                       |
|-----------|---------------------------------------------------------------------------------------------------|
| `name`    | Model name, default is `ViT-B/32`.                                                                |
| `providers`  | [ONNX runtime provides](https://onnxruntime.ai/docs/execution-providers/), default is auto-detect |

For example, to turn on JIT and force PyTorch running on CPU, one can do:

```{code-block} yaml
---
emphasize-lines: 9-11
---

jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      with: 
        jit: True
        device: cpu
      metas:
        py_modules:
          - executors/clip_torch.py
```

### Executor config

The full list of configs for Executor can be found via `jina executor --help`. The most important one is probably `replicas`, which **allows you to run multiple CLIP models in parallel** to achieve horizontal scaling.

To scale to 4 CLIP replicas, simply adding `replicas: 4` under `uses:`:

```{code-block} yaml
---
emphasize-lines: 9
---

jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      replicas: 4
      metas:
        py_modules:
          - executors/clip_torch.py
```


### Flow config

Flow configs are the ones under top-level `with:`. We can see the `port: 51000` is configured there. Besides `port`, there are some common parameters you might need.

| Parameter | Description                                                                                                                                                                                                           |
| --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `protocol` | Communication protocol between server and client.  Can be `grpc`, `http`, `websocket`.                                                                                                                                | 
| `cors`| Only effective when `protocol=http`. If set, a CORS middleware is added to FastAPI frontend to allow cross-origin access.                                                                                             |
| `prefetch` | Control the maximum streamed request inside the Flow at any given time, default is `None`, means no limit. Setting `prefetch` to a small number helps solving the OOM problem, but may slow down the streaming a bit. | 


As an example, to set `protocol` and `prefetch`, one can modify the YAML as follows:

```{code-block} yaml
---
emphasize-lines: 5,6
---

jtype: Flow
version: '1'
with:
  port: 51000
  protocol: websocket
  prefetch: 10
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      replicas: 4
      metas:
        py_modules:
          - executors/clip_torch.py
```

## Environment variables


To start a server with more verbose logging,

```bash
JINA_LOG_LEVEL=DEBUG python -m clip_server
```

```{figure} images/server-log.gif
:width: 70%

```

To run CLIP-server on 3rd GPU,

```bash
CUDA_VISIBLE_DEVICES=2 python -m clip_server
```