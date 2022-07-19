# Fine-tune Models

Although CLIP-as-service has provided you a list of pre-trained models, you can also train your models. 
This guide will show you how to use [Finetuner](https://finetuner.jina.ai) to fine-tune models and use them in CLIP-as-service.

For installation and basic usage of Finetuner, please refer to [Finetuner documentation](https://finetuner.jina.ai).
You can also [learn more details about fine-tuning CLIP](https://finetuner.jina.ai/tasks/text-to-image/).

## Prepare Training Data

Finetuner accepts training data and evaluation data in the form of [`DocumentArray`](https://docarray.jina.ai/fundamentals/documentarray/).
The training data for CLIP is a list of (text, image) pairs.
Each pair is store in a [`Document`](https://docarray.jina.ai/fundamentals/document/) which wraps two [`chunks`](https://docarray.jina.ai/fundamentals/document/nested/) with the `image` and `text` modality.
You can push the resulting [`DocumentArray`](https://docarray.jina.ai/fundamentals/documentarray/) to the cloud using the [`.push`](https://docarray.jina.ai/api/docarray.array.document/?highlight=push#docarray.array.document.DocumentArray.push) method.

We use [fashion captioning dataset](https://github.com/xuewyang/Fashion_Captioning) as a sample dataset in this tutorial.
You can get the description and image url from the dataset: 

| Description                                                                                                                           | Image URL                                                                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| subtly futuristic and edgy this liquid metal cuff bracelet is shaped from sculptural rectangular link                                 | [https://n.nordstrommedia.com/id/sr3/<br/>58d1a13f-b6b6-4e68-b2ff-3a3af47c422e.jpeg](https://n.nordstrommedia.com/id/sr3/58d1a13f-b6b6-4e68-b2ff-3a3af47c422e.jpeg) |
| high quality leather construction defines a hearty boot one-piece on a tough lug sole                                                 | [https://n.nordstrommedia.com/id/sr3/<br/>21e7a67c-0a54-4d09-a4a4-6a0e0840540b.jpeg](https://n.nordstrommedia.com/id/sr3/21e7a67c-0a54-4d09-a4a4-6a0e0840540b.jpeg) |
| this shimmering tricot knit tote is traced with decorative whipstitching and diamond cut chain the two hallmark of the falabella line | [https://n.nordstrommedia.com/id/sr3/<br/>1d8dd635-6342-444d-a1d3-4f91a9cf222b.jpeg](https://n.nordstrommedia.com/id/sr3/1d8dd635-6342-444d-a1d3-4f91a9cf222b.jpeg) |
| ...                                                                                                                                   | ...                                                                                                                                                                 |

You can use the following script to transform the first three entries of the dataset to a [`DocumentArray`](https://docarray.jina.ai/fundamentals/documentarray/) and push it to the cloud using the name `fashion-sample`.

```python
from docarray import Document, DocumentArray

train_da = DocumentArray(
    [
        Document(
            chunks=[
                Document(
                    content='subtly futuristic and edgy this liquid metal cuff bracelet is shaped from sculptural rectangular link',
                    modality='text',
                ),
                Document(
                    uri='https://n.nordstrommedia.com/id/sr3/58d1a13f-b6b6-4e68-b2ff-3a3af47c422e.jpeg',
                    modality='image',
                ),
            ],
        ),
        Document(
            chunks=[
                Document(
                    content='high quality leather construction defines a hearty boot one-piece on a tough lug sole',
                    modality='text',
                ),
                Document(
                    uri='https://n.nordstrommedia.com/id/sr3/21e7a67c-0a54-4d09-a4a4-6a0e0840540b.jpeg',
                    modality='image',
                ),
            ],
        ),
        Document(
            chunks=[
                Document(
                    content='this shimmering tricot knit tote is traced with decorative whipstitching and diamond cut chain the two hallmark of the falabella line',
                    modality='text',
                ),
                Document(
                    uri='https://n.nordstrommedia.com/id/sr3/1d8dd635-6342-444d-a1d3-4f91a9cf222b.jpeg',
                    modality='image',
                ),
            ],
        ),
    ]
)
train_da.push('fashion-sample')
```

The full dataset has been converted to `clip-fashion-train-data` and `clip-fashion-eval-data` and pushed to the cloud.
You can directly use them in Finetuner.

## Start Finetuner

You may now create and run a fine-tuning job after login to Jina ecosystem.

```python
import finetuner

finetuner.login()
run = finetuner.fit(
    model='openai/clip-vit-base-patch32',
    run_name='clip-fashion',
    train_data='clip-fashion-train-data',
    eval_data='clip-fashion-eval-data',  # optional
    epochs=5,
    learning_rate=1e-5,
    loss='CLIPLoss',
    cpu=False,
)
```

After the job started, you may use [`.status`](https://finetuner.jina.ai/api/finetuner.run/#finetuner.run.Run.status) to check the status of the job.

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

* First create a new folder named `clip-fashion-cas` or anything you like. This will be the storage of the models to use in CLIP-as-service.

* Second copy and move `clip-fashion/models/clip-text/model.onnx` to `clip-fashion-cas` and rename it to `textual.onnx`.

* Similarly, copy and move `clip-fashion/models/clip-vision/model.onnx` to `clip-fashion-cas` and rename it to `visual.onnx`.

This is the expected structure of `clip-fashion-cas`:

```text
.
└── clip-fashion-cas/
    ├── textual.onnx
    └── visual.onnx
```

In order to use fine-tuned model, create a custom YAML file `finetuned_clip.yml` like below. Learn more about [Flow YAML configuration](https://docs.jina.ai/fundamentals/flow/yaml-spec/) and [`clip_server` YAML configuration](https://clip-as-service.jina.ai/user-guides/server/#yaml-config).

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
          - clip_server.executors.clip_onnx
      with:
        name: ViT-B/32
        model_path: 'clip-fashion-cas' # path to clip-fashion-cas
    replicas: 1
```

```{warning}
Note that Finetuner only support ViT-B/32 CLIP model currently. The model name should match the fine-tuned model, or you will get incorrect output.
```

You can now start the `clip_server` using fine-tuned model to get a performance boost:

```bash
python -m clip_server finetuned_clip.yml
```

That's it, enjoy 🚀
