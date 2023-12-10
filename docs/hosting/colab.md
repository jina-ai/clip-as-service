# Host on Google Colab

```{figure} https://clip-as-service.jina.ai/_images/colab-banner.png
:width: 0 %
:scale: 0 %
```

```{figure} colab-banner.png
:scale: 0 %
:width: 0 %
```


As [Jina is fully compatible to Google Colab](https://docs.jina.ai/how-to/google-colab/), CLIP-as-service can be run smoothly on Colab as well. One can host `clip_server` on Google Colab by leveraging its free GPU/TPU resources and open up to 4 replicas of `ViT-L/14-336px`. Then you can send request from local to the server for embedding, ranking and reasoning tasks. 

Specifically, the architecture is illustrated below:

```{figure} cas-on-colab.svg
:width: 70%
```

```{button-link} https://colab.research.google.com/github/jina-ai/clip-as-service/blob/main/docs/hosting/cas-on-colab.ipynb
:color: primary
:align: center

{octicon}`link-external` Open the notebook on Google Colab 
```

Please follow the walk-through there. Enjoy the free GPU/TPU to build your awesome Jina applications!


```{tip}
Hosing service on Google Colab is not recommended if you server aims to be long-live or permanent. It is often used for quick experiment, demonstration or leveraging its free GPU/TPU. For stable, please deploy clip model on your own server.
```



