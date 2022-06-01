# Deploy CLIP Encoder on JCloud

This tutorial shows how to deploy CLIP Encoder on JCloud. 
Click [here](https://github.com/jina-ai/jcloud) for more information on JCloud usage.

## Deploy on JCloud

A sample YAML file of a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) to run minimum CLIP Encoder is given as

```yaml
# flow.yml
jtype: Flow
executors:
  - name: CLIPTorchEncoder # The name of the encoder
    uses: jinahub+docker://CLIPTorchEncoder # You can find more on Jina Hub
```

```{warning}
All Executors' `uses` must follow the format `jinahub+docker://MyExecutor` (from [Jina Hub](https://hub.jina.ai)) to avoid any local file dependencies.
```

To deploy,

```bash
$ jc deploy flow.yml
```

you should get:

```bash
╭───────────────Flow is available!──────────────╮
│                                               │
│   ID    <your_flow_id>                        │
│   URL   grpcs://<your_flow_id>.wolf.jina.ai   │
│                                               │
╰───────────────────────────────────────────────╯
```

## Connect from Client

Run the following Python script:

```python
from clip_client import Client

c = Client('grpcs://<your_flow_id>.wolf.jina.ai') # This is the URL you get from previous step

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

```bash
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