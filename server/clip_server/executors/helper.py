from typing import Tuple, List, Callable, TYPE_CHECKING
import numpy as np
from clip_server.model import clip

if TYPE_CHECKING:
    from docarray import Document, DocumentArray


def preproc_image(
    da: 'DocumentArray',
    preprocess_fn: Callable,
    device: str = 'cpu',
    return_np: bool = False,
) -> 'DocumentArray':
    for d in da:
        if d.blob:
            d.convert_blob_to_image_tensor()
        elif d.tensor is None and d.uri:
            # in case user uses HTTP protocol and send data via curl not using .blob (base64), but in .uri
            d.load_uri_to_image_tensor()

        d.tensor = preprocess_fn(d.tensor).detach()

    if return_np:
        da.tensors = da.tensors.cpu().numpy().astype(np.float32)
    else:
        da.tensors = da.tensors.to(device)
    return da


def preproc_text(
    da: 'DocumentArray', device: str = 'cpu', return_np: bool = False
) -> Tuple['DocumentArray', List[str]]:
    texts = da.texts
    da.tensors = clip.tokenize(texts).detach()

    if return_np:
        da.tensors = da.tensors.cpu().numpy().astype(np.int64)
    else:
        da.tensors = da.tensors.to(device)

    da[:, 'mime_type'] = 'text'
    return da, texts


def split_img_txt_da(doc: 'Document', img_da: 'DocumentArray', txt_da: 'DocumentArray'):
    if doc.text:
        txt_da.append(doc)
    elif doc.blob or (doc.tensor is not None):
        img_da.append(doc)
    elif doc.uri:
        img_da.append(doc)
