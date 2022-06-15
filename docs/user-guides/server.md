# Server API

CLIP-as-service is designed in a client-server architecture. A server is a long-running program that receives raw sentences and images from clients, and returns CLIP embeddings to the client. Additionally, `clip_server` is optimized for speed, low memory footprint and scalability.
- Horizontal scaling: adding more replicas easily with one argument. 
- Vertical scaling: using PyTorch JIT, ONNX or TensorRT runtime to speedup single GPU inference.
- Supporting gRPC, HTTP, Websocket protocols with their TLS counterparts, w/o compressions.

This chapter introduces the API of the server. 

```{tip}
You will need to install server first in Python 3.7+: `pip install clip-server`.
```

## Start server


### Start a PyTorch-backed server

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

### Start a ONNX-backed server

To use ONNX runtime for CLIP, you can run:

```bash
pip install "clip_server[onnx]"

python -m clip_server onnx-flow.yml
```


### Start a TensorRT-backed server

`nvidia-pyindex` package needs to be installed first. It allows your `pip` to fetch additional Python modules from the NVIDIA NGC™ PyPI repo:

```bash
pip install nvidia-pyindex
pip install "clip_server[tensorrt]"

python -m clip_server tensorrt-flow.yml
```

One may wonder where is this `onnx-flow.yml` or `tensorrt-flow.yml` come from. Must be a typo? Believe me, just run it. It should just work. I will explain this YAML file in the next section. 

The procedure and UI of ONNX and TensorRT runtime would look the same as Pytorch runtime.

## Model support

Open AI has released 9 models so far. `ViT-B/32` is used as default model in all runtimes. Due to the limitation of some runtime, not every runtime supports all nine models. Please also note that different model give different size of output dimensions. This will affect your downstream applications. For example, switching the model from one to another make your embedding incomparable, which breaks the downstream applications. Below is a list of supported models of each runtime and its corresponding size. We include the disk usage (in delta) and the peak RAM and VRAM usage (in delta) when running on a single Nvidia TITAN RTX GPU (24GB VRAM) using a default `minibatch_size=32` in server and a default `batch_size=8` in client.

| Model          | PyTorch | ONNX | TensorRT | Output Dimension | Disk Usage (MB) | Peak RAM Usage (GB) | Peak VRAM Usage (GB) |
|----------------|---------|------|----------|------------------|-----------------|---------------------|----------------------|
| RN50           | ✅       | ✅    | ✅        | 1024             | 256             | 2.99                | 1.36                 |
| RN101          | ✅       | ✅    | ✅        | 512              | 292             | 3.51                | 1.40                 |
| RN50x4         | ✅       | ✅    | ✅        | 640              | 422             | 3.23                | 1.63                 |
| RN50x16        | ✅       | ✅    | ❌        | 768              | 661             | 3.63                | 2.02                 |
| RN50x64        | ✅       | ✅    | ❌        | 1024             | 1382            | 4.08                | 2.98                 |
| ViT-B/32       | ✅       | ✅    | ✅        | 512              | 351             | 3.20                | 1.40                 |
| ViT-B/16       | ✅       | ✅    | ✅        | 512              | 354             | 3.20                | 1.44                 |
| ViT-L/14       | ✅       | ✅    | ✅        | 768              | 933             | 3.66                | 2.04                 |
| ViT-L/14-336px | ✅       | ✅    | ❌        | 768              | 934             | 3.74                | 2.23                 |


## YAML config

You may notice that there is a YAML file in our last ONNX example. All configurations are stored in this file. In fact, `python -m clip_server` does **not support** any other argument besides a YAML file. So it is the only source of the truth of your configs. 

And to answer your doubt, `clip_server` has three built-in YAML configs as a part of the package resources. When you do `python -m clip_server` it loads the Pytorch config, and when you do `python -m clip_server onnx-flow.yml` it loads the ONNX config.
In the same way, when you do `python -m clip_server tensorrt-flow.yml` it loads the TensorRT config.

Let's look at these three built-in YAML configs:

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


````{tab} tensorrt-flow.yml

```yaml
jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_r
    uses:
      jtype: CLIPEncoder
      metas:
        py_modules:
          - executors/clip_tensorrt.py
```
````

Basically, each YAML file defines a [Jina Flow](https://docs.jina.ai/fundamentals/flow/). The complete Jina Flow YAML syntax [can be found here](https://docs.jina.ai/fundamentals/flow/yaml-spec/). General parameters of the Flow and Executor can be used here as well. But now we only highlight the most important parameters.

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

For all backends, you can set the following parameters via `with`:

| Parameter | Description                                                                                                                    |
|-----------|--------------------------------------------------------------------------------------------------------------------------------|
| `name`    | Model weights, default is `ViT-B/32`. Support all OpenAI released pretrained models.                                           |
| `num_worker_preprocess` | The number of CPU workers for image & text prerpocessing, default 4.                                                           | 
| `minibatch_size` | The size of a minibatch for CPU preprocessing and GPU encoding, default 64. Reduce the size of it if you encounter OOM on GPU. |

There are also runtime-specific parameters listed below:

````{tab} PyTorch

| Parameter | Description                                                                                                                    |
|-----------|--------------------------------------------------------------------------------------------------------------------------------|
| `device`  | `cuda` or `cpu`. Default is `None` means auto-detect.                                                                          |
| `jit` | If to enable Torchscript JIT, default is `False`.                                                                              | 

````

````{tab} ONNX

| Parameter | Description                                                                                                                    |
|-----------|--------------------------------------------------------------------------------------------------------------------------------|
| `device`  | `cuda` or `cpu`. Default is `None` means auto-detect.

````

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
emphasize-lines: 7
---
jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    replicas: 4
    uses:
      jtype: CLIPEncoder
      metas:
        py_modules:
          - executors/clip_torch.py
```

(flow-config)=
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
    replicas: 4
    uses:
      jtype: CLIPEncoder
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

### Serve on Multiple GPUs

If you have multiple GPU devices, you can leverage them via `CUDA_VISIBLE_DEVICES=RR`. For example, if you have 3 GPUs and your Flow YAML says `replicas: 5`, then 

```bash
CUDA_VISIBLE_DEVICES=RR python -m clip_server
```

Will assign GPU devices to the following round-robin fashion:

| GPU device | Replica ID |
|------------|------------|
| 0          | 0          |
| 1          | 1          |
| 2          | 2          |
| 0          | 3          |
| 1          | 4          |


You can also restrict the visible devices in round-robin assigment by `CUDA_VISIBLE_DEVICES=RR0:2`, where `0:2` has the same meaning as Python slice. This will create the following assigment:

| GPU device | Replica ID |
|------------|------------|
| 0          | 0          |
| 1          | 1          |
| 0          | 2          |
| 1          | 3          |
| 0          | 4          |


```{tip}
In pratice, we found it is unnecessary to run `clip_server` on multiple GPUs for two reasons:
- A single replica even with largest `ViT-L/14-336px` takes only 3.5GB VRAM.
- Real network traffic never utilizes GPU in 100%.

Based on these two points, it makes more sense to have multiple replicas on a single GPU comparing to have multiple replicas on different GPU, which is kind of waste of resources. `clip_server` scales pretty well by interleaving the GPU time with mulitple replicas.
```

## Monitor with Prometheus and Grafana

To monitor the performance of the service, you can enable the Prometheus metrics in the Flow YAML:

```{code-block} yaml
---
emphasize-lines: 5,6,14,15
---

jtype: Flow
version: '1'
with:
  port: 51000
  monitoring: True
  port_monitoring: 9090
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      metas:
        py_modules:
          - executors/clip_torch.py
    monitoring: true
    port_monitoring: 9091
```

This enables Prometheus metrics on both Gateway and the CLIP Executor.

Running it gives you:

```{figure} images/server-start-monitoring.gif
:width: 80%

```

which exposes two additional endpoints:
- `http://localhost:9090`  for the Gateway
- `http://localhost:9091`  for the CLIP Executor


To visualize the metrics in Grafana, you can import this [JSON file of an example dashboard](https://clip-as-service.jina.ai/_static/cas-grafana.json). You will get something as follows:

```{figure} images/grafana-dashboard.png
:width: 80%

```


For more information on monitoring a Flow, [please read here](https://docs.jina.ai/fundamentals/flow/monitoring-flow/). 

## Serve with TLS

You can turn on TLS for HTTP and gRPC protocols. Your Flow YAML should be changed to the following:

```{code-block} yaml
---
emphasize-lines: 4,5,7-10
---
jtype: Flow
version: '1'
with:
  port: 8443
  protocol: http
  cors: true
  uvicorn_kwargs:
    ssl_keyfile_password: blahblah
  ssl_certfile: cert.pem
  ssl_keyfile: key.pem
```

Here, `protocol` can be either `http` or `grpc`; `cert.pem` or `key.pem` represent both parts of a certificate, key being the private key to the certificate and crt being the signed certificate. You can run the following command in terminal:

```bash
openssl req -newkey rsa:4096 -nodes -sha512 -x509 -days 3650 -nodes -out cert.pem -keyout key.pem -subj "/CN=demo-cas.jina.ai"
```

Note that if you are using `protocol: grpc` then `/CN=demo-cas.jina.ai` must strictly follow the IP address or the domain name of your server. Mismatch IP or domain name would throw an exception.

Certificate and keys can be also generated via [letsencrypt.org](https://letsencrypt.org/), which is a free SSL provider.

```{warning}
Note that note every port support HTTPS. Commonly support ports are: `443`, `2053`, `2083`, `2087`, `2096`, `8443`.
```

```{warning}
If you are using Cloudflare proxied DNS, please be aware:
- you need to turn on gRPC support manually, [please follow the guide here](https://support.cloudflare.com/hc/en-us/articles/360050483011-Understanding-Cloudflare-gRPC-support);
- the free tier of Cloudflare has 100s hard limit on the timeout, meaning sending big batch to a CPU server may throw 524 to the client-side.
```

When the server is successfully running, you can connect to it via client by setting `server` to `https://` or `grpcs://` as follows:

```python
from clip_client import Client

c = Client('grpcs://demo-cas.jina.ai:2096')

r = c.encode(
    [
        'First do it',
        'then do it right',
        'then do it better',
        'https://picsum.photos/200',
    ]
)
```

## Deploy on JCloud

The `clip_server` can be smoothly deployed and hosted as a [Flow](https://docs.jina.ai/fundamentals/flow/) on [JCloud](https://docs.jina.ai/fundamentals/jcloud/) to utilize the free computational and storage resources provided by Jina.

You need a YAML file to config the `clip_server` executor in the Flow in order to deploy. 
The executors are hosted on [Jina Hub](https://hub.jina.ai) and are sync with `clip_server` Python module. 
We currently support [PyTorch-backed CLIP](https://hub.jina.ai/executor/gzpbl8jh) and [ONNX-backed CLIP](https://hub.jina.ai/executor/2a7auwg2).

A minimum YAML file is as follows:

````{tab} pytorch-flow.yml

```yaml
jtype: Flow
executors:
  - name: CLIPTorchEncoder
    uses: jinahub+docker://CLIPTorchEncoder
```
````
````{tab} onnx-flow.yml

```yaml
jtype: Flow
executors:
  - name: CLIPOnnxEncoder
    uses: jinahub+docker://CLIPOnnxEncoder
```
````


```{warning}
All Executors' `uses` must follow the format `jinahub+docker://MyExecutor` (from [Jina Hub](https://hub.jina.ai)) to avoid any local file dependencies.
```

To deploy,

````{tab} PyTorch-backed
```bash
$ jc deploy pytorch-flow.yml
```
````
````{tab} ONNX-backed
```bash
$ jc deploy onnx-flow.yml
```
````

Here `jc deploy` is the command to deploy a Jina project to JCloud.
Learn more about [JCloud usage](https://github.com/jina-ai/jcloud).

The Flow is successfully deployed when you see:

```{figure} images/jc-deploy.png
:width: 60%
```

After deploying on jcloud, you can connect to it via client by setting  `grpcs://` generated from previous step as follows:

```python
from clip_client import Client

c = Client(
    'grpcs://174eb69ba3.wolf.jina.ai'
)  # This is the URL you get from previous step

r = c.encode(
    [
        'First do it',
        'then do it right',
        'then do it better',
        'https://picsum.photos/200',
    ]
)
print(r)
```

will give you:

```text
[[ 0.03480401 -0.23519686  0.01041038 ... -0.5229086  -0.10081214
   -0.08695138]
 [-0.0683605  -0.00324154  0.01490371 ... -0.50309485 -0.06193433
   -0.08574048]
 [ 0.15041807 -0.07933374 -0.06650036 ... -0.46410388 -0.08535041
   0.04270519]
 [-0.16183889  0.10636599 -0.2062868  ... -0.41244072  0.19485454
   0.05658712]]
```

It means the client and the JCloud server are now connected. Well done!