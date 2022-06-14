# Text & Image Embedding

Embedding is a basic task in CLIP-as-service. It means converting your input sentence or image into a fixed-length vector. In this demo, you can choose a picture, input a sentence in the textbox, or copy-paste your image URL into the text box to get a rough feeling how CLIP-as-service works.

This is *not* a search task. The images are random stock images and are related to any search results, they are mainly for saving your time on finding some random internet cat pictures. 

The model is `ViT-L/14-336px` on one GPU.

<iframe frameborder="0" allowtransparency="true" scrolling="no" src="../../_static/demo-embed.html" style="overflow:hidden;overflow-x:hidden;overflow-y:hidden;height:100vh;width:100%"></iframe>

```{button-link} ../../_static/demo-text-rank.html
:color: primary
:align: center

{octicon}`link-external` Open this playground in a new window
```