# Welcome to CLIP-as-service!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

## Try it!

An always-online demo server loaded with `ViT-L/14-336px` is there for you to play & test: 

````{tab} via HTTPS 🔐

```bash
curl \
-X POST https://demo-cas.jina.ai:8443/post \
-H 'Content-Type: application/json' \
-d '{"data":[{"text": "First do it"}, 
    {"text": "then do it right"}, 
    {"text": "then do it better"}, 
    {"uri": "https://picsum.photos/200"}], 
    "execEndpoint":"/"}'
```

````

````{tab} via gRPC ⚡⚡

```bash
pip install clip-client
```

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
print(r)
```

````

## Install

![PyPI](https://img.shields.io/pypi/v/clip_client?color=%23ffffff&label=%20) is the latest version.

Make sure you have Python 3.7+. You can install client and server independently. You **don't** have to install both: e.g. installing `clip_server` on a GPU machine and `clip_client` on a local laptop.

````{tab} Client

```bash
pip install clip-client
```

````

````{tab} Server (PyTorch)

```bash
pip install clip-server
```
````

````{tab} Server (ONNX)

```bash
pip install "clip_server[onnx]"
```

````


````{tab} Server (TensorRT)

```bash
pip install nvidia-pyindex 
pip install "clip_server[tensorrt]"
```
````

````{tab} Server on Google Colab

```{button-link} https://colab.research.google.com/github/jina-ai/clip-as-service/blob/main/docs/hosting/cas-on-colab.ipynb
:color: primary
:align: center

{octicon}`link-external` Open the notebook on Google Colab 
```

````



## Quick check

After install, you can run the following commands for a quick connectivity check.

### Start the server

````{tab} Run PyTorch Server 
```bash
python -m clip_server
```
````

````{tab} Run ONNX Server 
```bash
python -m clip_server onnx-flow.yml
```
````

````{tab} Run TensorRT Server 
```bash
python -m clip_server tensorrt-flow.yml
```
````

At the first time, it will download the default pretrained model, which may take a minute. Then you will get the following address information: 

```text
 🔗         Protocol                  GRPC   
 🏠     Local access         0.0.0.0:51000   
 🔒  Private network    192.168.3.62:51000   
 🌐   Public address  87.105.159.191:51000   
```

It means the server is ready to serve. Note down the three addresses showed above, you will need them later.

### Connect from client

```{tip}
Depending on the location of the client and server. You may use different IP addresses:
- Client and server are on the same machine: use local address e.g. `0.0.0.0`
- Client and server are behind the same router: use private network address e.g. `192.168.3.62`
- Server is in public network: use public network address e.g. `87.105.159.191`
```

Run the following Python script:

```python
from clip_client import Client

c = Client('grpc://0.0.0.0:51000')
c.profile()
```

will give you:

```text
 Roundtrip  16ms  100%                                                          
├──  Client-server network  12ms  75%                                           
└──  Server  4ms  25%                                                           
    ├──  Gateway-CLIP network  0ms  0%                                          
    └──  CLIP model  4ms  100%      
```

It means the client and the server are now connected. Well done!


```{include} ../README.md
:start-after: <!-- start support-pitch -->
:end-before: <!-- end support-pitch -->
```


```{toctree}
:caption: User Guides
:hidden:

user-guides/client
user-guides/server
user-guides/retriever
user-guides/faq
```

```{toctree}
:caption: Hosting
:hidden:

hosting/colab
hosting/on-jcloud
```

```{toctree}
:caption: Playground
:hidden:

playground/embedding
playground/reasoning
```


```{toctree}
:caption: Developer References
:hidden:
:maxdepth: 1

api/clip_client
changelog/index
```


---
{ref}`genindex` | {ref}`modindex`

