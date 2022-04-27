import os

import pytest
from docarray import Document, DocumentArray
from jina import Flow

from clip_client.client import Client


@pytest.mark.gpu
@pytest.mark.parametrize(
    'inputs',
    [
        [Document(text='hello, world'), Document(text='goodbye, world')],
        DocumentArray([Document(text='hello, world'), Document(text='goodbye, world')]),
        lambda: (Document(text='hello, world') for _ in range(10)),
        DocumentArray(
            [
                Document(uri='https://docarray.jina.ai/_static/favicon.png'),
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/img/00000.jpg'
                ),
                Document(text='hello, world'),
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/img/00000.jpg'
                ).load_uri_to_image_tensor(),
            ]
        ),
        DocumentArray.from_files(
            f'{os.path.dirname(os.path.abspath(__file__))}/**/*.jpg'
        ),
    ],
)
def test_docarray_inputs(make_trt_flow, inputs):
    c = Client(server=f'grpc://0.0.0.0:{make_trt_flow.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert isinstance(r, DocumentArray)
    assert r.embeddings.shape
