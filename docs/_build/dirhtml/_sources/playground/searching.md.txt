# Text & Image Searching

CLIP-as-service enables us to encode text and images into a common space. This is a powerful tool for many applications, such as cross-modality search.

[CLIP search](../user-guides/retriever.md) is a new feature provided by CLIP-as-service. It enables us to search for images based on text/image. It calculates the similarity score based on the embeddings of the text and image. The higher the score, the more similar they are.

This demo demonstrates the text-to-image and image-to-image searching in CLIP search. You can type text query or upload the local image as a query, and it will return the top 10 similar images for you.

In this demo, we use [``Open-Image-Dataset``](https://storage.googleapis.com/openimages/web/download.html) dataset (consist of 125,346 images) to demonstrate Text & Image retrieval.

<iframe frameborder="0" allowtransparency="true" scrolling="no" src="https://jemmyshin-laion5b-streamlit-streamlit-demo-rddbqz.streamlitapp.com?embedded=true" style="overflow:hidden;overflow-x:hidden;overflow-y:hidden;height:100vh;width:100%"></iframe>

```{button-link} https://jemmyshin-laion5b-streamlit-streamlit-demo-rddbqz.streamlitapp.com/
:color: primary
:align: center

{octicon}`link-external` Open this playground in a new window
```