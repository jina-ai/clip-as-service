# Fine-tuning CLIP Models Using Finetuner

Although CLIP-as-service has provided you a list of pre-trained models, you can also train your own models. 
This guide will show you how to use [Finetuner](https://finetuner.jina.ai) to fine-tune models and use them in CLIP-as-service.

For installation and basic usage of Finetuner, please refer to [Finetuner website](https://finetuner.jina.ai).

## Prepare Training Data

Finetuner accepts training data and evaluation data in the form of `DocumentArray`.
The training data for CLIP is a list of (text, image) pairs.
Each pair is store in a `Document` which wraps two `chunk`s with the `image` and `text` modality.
You can push the resulting `DocumentArray` to the cloud using the `.push` method.
A sample to construct and push the training data is shown below.

[//]: # ()
[//]: # (```python)

[//]: # (from docarray import Document, DocumentArray)

[//]: # ()
[//]: # (train_da = DocumentArray&#40;)

[//]: # (    [)

[//]: # (        Document&#40;)

[//]: # (            chunks=[)

[//]: # (                Document&#40;)

[//]: # (                    content='pencil skirt slim fit available for sell',)

[//]: # (                    modality='text',)

[//]: # (                &#41;,)

[//]: # (                Document&#40;)

[//]: # (                    uri='https://...skirt-1.png',)

[//]: # (                    modality='image',)

[//]: # (                &#41;,)

[//]: # (            ],)

[//]: # (        &#41;,)

[//]: # (    ])

[//]: # (&#41;)

[//]: # (train_da.push&#40;'clip-fashion-train-data'&#41;)

[//]: # (```)

[//]: # ()
[//]: # (## Run Job)

[//]: # ()
[//]: # (You may now create and run a fine-tuning job after login to Jina ecosystem.)

[//]: # ()
[//]: # (```python)

[//]: # (import finetuner)

[//]: # ()
[//]: # (finetuner.login&#40;&#41;)

[//]: # ()
[//]: # (run = finetuner.fit&#40;)

[//]: # (    model='openai/clip-vit-base-patch32',)

[//]: # (    run_name='clip-fashion',)

[//]: # (    train_data='clip-fashion-train-data',)

[//]: # (    eval_data='clip-fashion-eval-data',)

[//]: # (    epochs=5,)

[//]: # (    learning_rate=1e-5,)

[//]: # (    loss='CLIPLoss',)

[//]: # (    cpu=False,)

[//]: # (&#41;)

[//]: # (```)

[//]: # ()
[//]: # (After the job started, you may use `.status` to check the status of the job.)

[//]: # ()
[//]: # (```python)

[//]: # (import finetuner)

[//]: # ()
[//]: # (finetuner.login&#40;&#41;)

[//]: # (run = finetuner.get_run&#40;'clip-fashion'&#41;)

[//]: # (print&#40;run.status&#40;&#41;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (When the status is `FINISHED`, you can download the tuned model to your local machine.)

[//]: # ()
[//]: # (```python)

[//]: # (import finetuner)

[//]: # ()
[//]: # (finetuner.login&#40;&#41;)

[//]: # (run = finetuner.get_run&#40;'clip-fashion'&#41;)

[//]: # (run.save_artifact&#40;'clip-model'&#41;)

[//]: # (```)

[//]: # ()
[//]: # (You should now get a zip file containing the tuned model named `clip-fashion.zip` under the folder `clip-model`.)

[//]: # ()
[//]: # (## Use the Model)

[//]: # ()
[//]: # (After unzipping the model you get from the previous step, a folder with the following structure will be generated:)

[//]: # ()
[//]: # (```text)

[//]: # (.)

[//]: # (└── clip-fashion/)

[//]: # (    ├── config.yml)

[//]: # (    ├── metadata.yml)

[//]: # (    ├── metrics.yml)

[//]: # (    └── models/)

[//]: # (        ├── clip-text/)

[//]: # (        │   ├── metadata.yml)

[//]: # (        │   └── model.onnx)

[//]: # (        ├── clip-vision/)

[//]: # (        │   ├── metadata.yml)

[//]: # (        │   └── model.onnx)

[//]: # (        └── input-map.yml)

[//]: # (```)

[//]: # ()
[//]: # (Since the tuned model generated from Finetuner contains richer information such as metadata and config, we now transform it to simpler structure used by CLIP-as-service.)

[//]: # ()
[//]: # (First create another folder named `clip-fashion-cas` or anything you like, this will be the storage of the models to use in CLIP-as-service.)

[//]: # ()
[//]: # (Then copy and move `clip-fashion/models/clip-text/model.onnx` to `clip-fashion-cas` and rename it to `textual.onnx`.)

[//]: # ()
[//]: # (Similarly, copy and move `clip-fashion/models/clip-vision/model.onnx` to `clip-fashion-cas` and rename it to `visual.onnx`.)

[//]: # ()
[//]: # (Now that you should have your clip-fashion-cas structured like this:)

[//]: # ()
[//]: # (```text)

[//]: # (.)

[//]: # (└── clip-fashion-cas/)

[//]: # (    ├── textual.onnx)

[//]: # (    └── visual.onnx)

[//]: # (```)

[//]: # ()
[//]: # (In order to use finetuned model, create a custom yaml file `finetuned_clip.yml`. For more information on flow and `clip_server` customization, please refer to [https://docs.jina.ai/fundamentals/flow/yaml-spec/]&#40;https://docs.jina.ai/fundamentals/flow/yaml-spec/&#41; and [https://clip-as-service.jina.ai/user-guides/server/#yaml-config]&#40;https://clip-as-service.jina.ai/user-guides/server/#yaml-config&#41;)

[//]: # ()
[//]: # (```yaml)

[//]: # (jtype: Flow)

[//]: # (version: '1')

[//]: # (with:)

[//]: # (  port: 51000)

[//]: # (executors:)

[//]: # (  - name: clip_o)

[//]: # (    uses:)

[//]: # (      jtype: CLIPEncoder)

[//]: # (      metas:)

[//]: # (        py_modules:)

[//]: # (          - executors/clip_onnx.py)

[//]: # (      with:)

[//]: # (        name: ViT-B/32 # since finetuner only support ViT-B/32 for CLIP)

[//]: # (        model_path: 'clip-fashion-cas' # path to clip-fashion-cas)

[//]: # (    replicas: 1)

[//]: # (```)

[//]: # ()
[//]: # (You can now start the `clip_server` using fine-tuned model to get a performance boost:)

[//]: # ()
[//]: # (```bash)

[//]: # (python -m clip_server finetuned_clip.yml)

[//]: # (```)

[//]: # ()
[//]: # (That's it! )