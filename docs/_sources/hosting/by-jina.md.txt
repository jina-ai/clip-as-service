# Hosted by Jina AI

```{include} ../../README.md
:start-after: <!-- start inference-banner -->
:end-before: <!-- end inference-banner -->
```

In today's dynamic business environment, enterprises face a multitude of challenges that require advanced solutions to 
maintain a competitive edge. 
From managing vast amounts of unstructured data to delivering personalized customer experiences, businesses need 
efficient tools to tackle these obstacles. 
Machine learning (ML) has emerged as a powerful tool for automating repetitive tasks, processing data effectively, and 
generating valuable insights from multimedia content. 
Jina AI's Inference offers a comprehensive solution to streamline access to curated, state-of-the-art ML models, 
eliminating traditional roadblocks such as costly and time-consuming MLOps steps and the distinction between public and 
custom neural network models.

## Getting started

To access the fastest and most performant CLIP models, [Jina AI's Inference](https://cloud.jina.ai/user/inference) is 
the go-to choice. 
Follow the steps below to get started:

1. Sign up for a free account at [Jina AI Cloud](https://cloud.jina.ai).
2. Once you have created an account, navigate to the Inference tab to create a new CLIP model.
3. The model can be accessed either through an HTTP endpoint or a gRPC endpoint.

## Obtaining a Personal Access Token

Before you begin using [Jina AI's Inference](https://cloud.jina.ai/user/inference), ensure that you have obtained a 
personal access token (PAT) from the [Jina AI Cloud](https://cloud.jina.ai) or through the command-line interface (CLI). 
Use the following guide to create a new PAT:

1. Access the [Jina AI Cloud](https://cloud.jina.ai) and log in to your account.
2. Navigate to the [**Access token**](https://cloud.jina.ai/settings/tokens) section in the **Settings** tab, or alternatively, create a PAT via the CLI using the command:

```bash
jina auth token create <name of PAT> -e <expiration days>
```

## Installing the Inference Client

To interact with the model created in Inference, you will need to install the `inference-client` Python package. 
Follow the steps below to install the package using pip:

```bash
pip install inference-client
```

## Interacting with the Model

Once you have your personal access token and the model name listed in the Inference detail page, you can start 
interacting with the model using the `inference-client` Python package. 
Follow the example code snippet below:

```python
from inference_client import Client

client = Client(token='<your auth token>')

model = client.get_model('<your model name>')
```

The CLIP models offer the following functionalities:

1. Encoding: Users can encode data by calling the `model.encode` method. For detailed instructions on using this method, refer to the [Encode documentation](https://jina.readme.io/docs/encode).
2. Ranking: Users can perform ranking by calling the `model.rank` method. Refer to the [Rank documentation](https://jina.readme.io/docs/rank) for detailed instructions on using this method.

For further details on usage and information about other tasks and models supported in Inference, as well as how to use 
`curl` to interact with the model, please consult the [Inference documentation](https://jina.readme.io/docs/inference).
