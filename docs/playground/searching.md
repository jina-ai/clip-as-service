# Text & Image Searching

CLIP-as-service enables us to encode text and images into a common space and perform cross-modality search. This is a powerful tool for many applications, such as cross-modality search.

[CLIP search](../user-guides/retriever.md) uses the CLIP-as-service to encode the text and the image. The text and the image are then used to calculate the similarity between them. The similarity score is used to sort the results.

This demo demonstrates the text-to-image and image-to-image searching in CLIP search. You can type text query or upload the local image as a query, and it will return the top 10 similar images for you.

For dataset used in this demo, we have used the [``Open-Image-Datset``](https://storage.googleapis.com/openimages/web/download.html) dataset. It contains 125,346 images in total.

<iframe frameborder="0" allowtransparency="true" scrolling="no" src="https://jemmyshin-laion5b-streamlit-streamlit-demo-rddbqz.streamlitapp.com?embedded=true" style="overflow:hidden;overflow-x:hidden;overflow-y:hidden;height:100vh;width:100%"></iframe>

```{button-link} https://jemmyshin-laion5b-streamlit-streamlit-demo-rddbqz.streamlitapp.com/
:color: primary
:align: center

{octicon}`link-external` Open this playground in a new window
```