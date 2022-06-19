# Host on JCloud

```{warning}
JCloud does not support GPU hosting yet. Hence `clip_server` deployed on JCloud will be run on CPU.
```

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

Note that, `port` is unnecessary here as JCloud will assign a new URL for any deployed service. 

Executors now must start with `jinahub+docker://` as it is required by JCloud. We currently provide containerized executors [`jinahub+docker://CLIPTorchEncoder`](https://hub.jina.ai/executor/gzpbl8jh) and [`jinahub+docker://CLIPOnnxEncoder`](https://hub.jina.ai/executor/2a7auwg2) on Jina Hub. They are automatically synced on the new release of `clip_server` module. 

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
