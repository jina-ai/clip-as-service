# Fine-tuning CLIP models using Finetuner

Although CLIP-as-service has provided you a list of pre-trained models, you can also train your own models. 
This guide will show you how to use [Finetuner](https://finetuner.jina.ai) to fine-tune models and use them in CLIP-as-service.

For installation and basic usage of Finetuner, please refer to [Finetuner website](https://finetuner.jina.ai).

## Prepare Training Data

Finetuner accepts training data and evaluation data in the form of `DocumentArray`.
The training data for CLIP is a list of (text, image) pairs.
Each pair is store in a `Document` which wraps two `chunk`s with the `image` and `text` modality.
You can push the resulting `DocumentArray` to the cloud using the `.push` method.
A sample to construct and push the training data is shown below.

```python
from docarray import Document, DocumentArray

train_da = DocumentArray(
    [
        Document(
            chunks=[
                Document(
                    content='pencil skirt slim fit available for sell',
                    modality='text',
                ),
                Document(
                    uri='https://...skirt-1.png',
                    modality='image',
                ),
            ],
        ),
    ]
)
train_da.push('clip-fashion-train-data')
```

## Run Job

You may now create and run a fine-tuning job after login to Jina ecosystem.

```python
import finetuner

finetuner.login()

run = finetuner.fit(
    model='openai/clip-vit-base-patch32',
    run_name='clip-fashion',
    train_data='clip-fashion-train-data',
    eval_data='clip-fashion-eval-data',
    epochs=5,
    learning_rate=1e-5,
    loss='CLIPLoss',
    cpu=False,
)
```

After the job started, you may use `.status` to check the status of the job.

```python
import finetuner

finetuner.login()
run = finetuner.get_run('clip-fashion')
print(run.status())
```

When the status is `FINISHED`, you can download the tuned model to your local machine.

```python
import finetuner

finetuner.login()
run = finetuner.get_run('clip-fashion')
run.save_artifact('clip-model')
```

You should now get a zip file containing the tuned model named `clip-fashion.zip` under the folder `clip-model`.

## Use the Model

After unzipping the model you get from the previous step, a folder with the following structure will be generated:

```text
.
└── clip-fashion/
    ├── config.yml
    ├── metadata.yml
    ├── metrics.yml
    └── models/
        ├── clip-text/
        │   ├── metadata.yml
        │   └── model.onnx
        ├── clip-vision/
        │   ├── metadata.yml
        │   └── model.onnx
        └── input-map.yml
```

Since the tuned model generated from Finetuner contains richer information such as metadata and config, we now transform it to simpler structure used by CLIP-as-service.

First create another folder named `clip-fashion-cas` or anything you like, this will be the storage of the models to use in CLIP-as-service.

Then copy and move `clip-fashion/models/clip-text/model.onnx` to `clip-fashion-cas` and rename it to `textual.onnx`.

Similarly, copy and move `clip-fashion/models/clip-vision/model.onnx` to `clip-fashion-cas` and rename it to `visual.onnx`.

Now that you should have your clip-fashion-cas structured like this:

```text
.
└── clip-fashion-cas/
    ├── textual.onnx
    └── visual.onnx
```

In order to use finetuned model, create a custom yaml file `finetuned_clip.yml`. For more information on flow and `clip_server` customization, please refer to [https://docs.jina.ai/fundamentals/flow/yaml-spec/](https://docs.jina.ai/fundamentals/flow/yaml-spec/) and [https://clip-as-service.jina.ai/user-guides/server/#yaml-config](https://clip-as-service.jina.ai/user-guides/server/#yaml-config)

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
      with:
        name: ViT-B/32 # since finetuner only support ViT-B/32 for CLIP
        model_path: 'clip-fashion-cas' # path to clip-fashion-cas
    replicas: 1
```

You can now start the `clip_server` using fine-tuned model to get a performance boost:

```bash
python -m clip_server finetuned_clip.yml
```

That's it! 