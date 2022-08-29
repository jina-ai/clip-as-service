# Host on JCloud

Essentially `clip_server` is a Jina [Flow](https://docs.jina.ai/fundamentals/flow/). Any Jina Flow can be hosted on [JCloud](https://docs.jina.ai/fundamentals/jcloud/), hence `clip_server` can be hosted on JCloud as well. Learn more about [JCloud here](https://docs.jina.ai/fundamentals/jcloud/).


First, you need a Flow YAML file for deploy. A minimum YAML file is as follows:

````{tab} torch-flow.yml

```yaml
jtype: Flow
executors:
  - uses: jinahub+docker://CLIPTorchEncoder
```

````
````{tab} onnx-flow.yml

```yaml
jtype: Flow
executors:
  - uses: jinahub+docker://CLIPOnnxEncoder
```

````

```{tip}
`port` is unnecessary here as JCloud will assign a new hostname and port for any deployed service. 
```

Executors must start with `jinahub+docker://` as it is required by JCloud. We currently provide containerized executors [`jinahub+docker://CLIPTorchEncoder`](https://hub.jina.ai/executor/gzpbl8jh) and [`jinahub+docker://CLIPOnnxEncoder`](https://hub.jina.ai/executor/2a7auwg2) on Jina Hub. They are automatically synced on the new release of `clip_server` module. 

To enable GPU on JCloud, you need to configure it in the YAML file and use prebuilt docker GPU images. For example,

```yaml
jtype: Flow
executors:
  - uses: jinahub+docker://CLIPTorchEncoder/0.5.2-gpu
    jcloud:
      resources:
        gpu: shared
```

Please refer [here](https://docs.jina.ai/fundamentals/jcloud/yaml-spec/#gpu) for more details on using GPU in JCloud.
Notice that you must specify a docker image GPU tag for your executor to utilize the GPU. For example `0.5.2-gpu`. 
See the 'Tag' section in [CLIPTorchEncoder](https://hub.jina.ai/executor/gzpbl8jh) and [CLIPOnnxEncoder](https://hub.jina.ai/executor/2a7auwg2) for docker image GPU tags.

To deploy,

````{tab} PyTorch-backed
```bash
jc deploy torch-flow.yml
```
````

````{tab} ONNX-backed
```bash
jc deploy onnx-flow.yml
```
````


If Flow is successfully deployed you will see:

```{figure} jc-deploy.png
:width: 60%
```

You can now connect to it via client by setting  `server` as the URL given by JCloud:

```python
from clip_client import Client

c = Client(
    'grpcs://174eb69ba3.wolf.jina.ai'
)  # This is the URL you get from previous step
c.profile()
```
