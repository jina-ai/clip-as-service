# Visual Reasoning

Visual reasoning is another basic task in CLIP-as-service. There are four basic visual reasoning skills: object recognition, object counting, color recognition, and spatial relation understanding. Despite how magic it sounds and looks, the idea is fairly simple: just input the reasoning texts as prompts, then {ref}`calling rank interface<rank-api>` of `clip_server`. The server will rank the prompts and return sorted prompts with scores.

In this demo, you can choose a picture, or copy-paste your image URL into the text box to get a rough feeling how visual reasoning works. Feel free to add or remove prompts and observe how it affects the ranking results.

The model is `ViT-L/14-336px` on one GPU.

<iframe frameborder="0" allowtransparency="true" scrolling="no" src="../../_static/demo-text-rank.html" style="overflow:hidden;overflow-x:hidden;overflow-y:hidden;height:100vh;width:100%"></iframe>


```{button-link} ../../_static/demo-text-rank.html
:color: primary
:align: center

{octicon}`link-external` Open this playground in a new window
```