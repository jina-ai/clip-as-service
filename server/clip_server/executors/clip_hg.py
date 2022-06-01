from copy import deepcopy
from typing import Any, Dict, Optional, Sequence

import torch
from docarray import DocumentArray
from jina import Executor, requests
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer


class CLIPEncoder(Executor):
    def __init__(
        self,
        pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32',
        finetuned_checkpoint_path: Optional[str] = None,
        base_feature_extractor: Optional[str] = None,
        base_tokenizer_model: Optional[str] = None,
        use_default_preprocessing: bool = True,
        max_length: int = 77,
        device: str = 'cpu',
        overwrite_embeddings: bool = False,
        traversal_paths: str = '@r',
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Can be either:
            - A string, the model id of a pretrained CLIP model hosted
                inside a model repo on huggingface.co, e.g.,
                'openai/clip-vit-base-patch32'
            - A path to a directory containing model weights saved, e.g.,
                ./my_model_directory/
        :param finetuned_checkpoint_path: If set, the pretrained model weights will be replaced with weights
            loading from the given checkpoint.
        :param base_feature_extractor: Base feature extractor for images.
            Defaults to ``pretrained_model_name_or_path`` if None.
        :param base_tokenizer_model: Base tokenizer model.
            Defaults to ``pretrained_model_name_or_path`` if None.
        :param use_default_preprocessing: Whether to use the `base_feature_extractor`
            on images (tensors) before encoding them. If you disable this, you must
            ensure that the images you pass in have the correct format, see the
            ``encode`` method for details.
        :param max_length: Max length argument for the tokenizer. All CLIP models
            use 77 as the max length.
        :param device: Pytorch device to put the model on, e.g. 'cpu', 'cuda',
            'cuda:1'.
        :param overwrite_embeddings: Whether to overwrite existing embeddings. By
            default docs that have embeddings already are not processed. This value
            can be overwritten if the same parameter is passed to the request.
        :param traversal_paths: Default traversal paths for encoding, used if
            the traversal path is not passed as a parameter with the request.
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        """
        super().__init__(*args, **kwargs)
        self.overwrite_embeddings = overwrite_embeddings
        self.traversal_paths = traversal_paths
        self.batch_size = batch_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.use_default_preprocessing = use_default_preprocessing
        self.base_feature_extractor = (
            base_feature_extractor or pretrained_model_name_or_path
        )
        self.max_length = max_length

        self.device = device
        self.vision_preprocessor = CLIPFeatureExtractor.from_pretrained(
            self.base_feature_extractor
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)

        if finetuned_checkpoint_path:
            if finetuned_checkpoint_path.startswith(
                'https://'
            ) or finetuned_checkpoint_path.startswith('http://'):
                state_dict = torch.hub.load_state_dict_from_url(
                    finetuned_checkpoint_path, map_location='cpu', progress=True
                )
            else:
                state_dict = torch.load(finetuned_checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)

        self.model.eval().to(device)

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict[str, Any], **_):
        """
        Encode all documents with `text` or image content using the corresponding CLIP
        encoder. Store the embeddings in the `embedding` attribute. Documents with
        existing embeddings are not processed unless `overwrite_embeddings` is set to
        True.
        :param docs: Documents sent to the encoder. The image docs must have
            ``tensor`` of the
            shape ``Height x Width x 3``. By default, the input ``tensor`` must
            be an ``ndarray`` with ``dtype=uint8`` or ``dtype=float32``.
            If you set ``use_default_preprocessing=True`` when creating this encoder,
            then the ``tensor`` arrays should have the shape ``[H, W, 3]``, and be in
            the RGB color format with ``dtype=uint8``.
            If you set ``use_default_preprocessing=False`` when creating this encoder,
            then you need to ensure that the images you pass in are already
            pre-processed. This means that they are all the same size (for batching) -
            the CLIP model was trained on images of the size ``224 x 224``, and that
            they are of the shape ``[3, H, W]``  with ``dtype=float32``. They should
            also be normalized (values between 0 and 1).
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``traversal_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """
        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        batch_size = parameters.get('batch_size', self.batch_size)
        overwrite_embeddings = parameters.get(
            'overwrite_embeddings', self.overwrite_embeddings
        )
        text_docs = DocumentArray(
            filter(
                lambda x: (
                    bool(x.text) and (overwrite_embeddings or x.embedding is None)
                ),
                docs[traversal_paths],
            )
        )
        image_docs = DocumentArray(
            filter(
                lambda x: (
                    (x.tensor is not None or x.blob != b'' or x.uri)
                    and (overwrite_embeddings or x.embedding is None)
                ),
                docs[traversal_paths],
            )
        )

        with torch.inference_mode():
            for batch in text_docs.batch(batch_size=batch_size):
                self._encode_texts(batch)

            for batch in image_docs.batch(batch_size=batch_size):
                self._encode_images(batch)

    def _encode_images(self, batch: DocumentArray):
        """Encode images using the CLIP image encoder."""
        tensors_batch = []
        for d in batch:
            if d.blob != b'':
                d_tmp = deepcopy(d)
                d_tmp.convert_blob_to_image_tensor()
                tensor = d_tmp.tensor
            elif d.tensor is not None:
                tensor = d.tensor
            else:
                # must be uri
                d_tmp = deepcopy(d)
                d_tmp.load_uri_to_image_tensor()
                tensor = d_tmp.tensor
            tensors_batch.append(tensor)
        if self.use_default_preprocessing:
            tensor = self._preprocess_images(tensors_batch)
        else:
            tensor = {
                'pixel_values': torch.tensor(
                    batch.tensors, dtype=torch.float32, device=self.device
                )
            }

        embeddings = self.model.get_image_features(**tensor)
        embeddings = embeddings.cpu().numpy()
        batch.embeddings = embeddings

    def _encode_texts(self, batch: DocumentArray):
        """Encode texts using the CLIP text encoder."""
        text_batch = batch.texts
        input_tokens = self._tokenize_texts(text_batch)
        embeddings = self.model.get_text_features(**input_tokens).cpu().numpy()
        for doc, embedding in zip(batch, embeddings):
            doc.embedding = embedding

    def _preprocess_images(self, images):
        """Preprocess images."""
        x = self.vision_preprocessor(
            images=images,
            return_tensors='pt',
        )
        return {k: v.to(torch.device(self.device)) for k, v in x.items()}

    def _tokenize_texts(self, texts: Sequence[str]):
        """Tokenize texts."""
        x = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        return {k: v.to(self.device) for k, v in x.items()}
