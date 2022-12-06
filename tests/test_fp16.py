import os

import pytest
from docarray import Document, DocumentArray
from jina import Flow

from clip_client.client import Client


@pytest.mark.gpu
@pytest.mark.parametrize(
    'inputs',
    [
        ['hello, world', 'goodbye, world'],
        ('hello, world', 'goodbye, world'),
        lambda: ('hello, world' for _ in range(10)),
        [
            'https://docarray.jina.ai/_static/favicon.png',
            f'{os.path.dirname(os.path.abspath(__file__))}/img/00000.jpg',
            'hello, world',
        ],
    ],
)
def test_plain_inputs(make_flow, inputs):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert (
        r.shape[0] == len(list(inputs)) if not callable(inputs) else len(list(inputs()))
    )


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
def test_docarray_inputs(make_flow, inputs):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert isinstance(r, DocumentArray)
    assert r.embeddings.shape
    assert not r[0].tensor
    if hasattr(inputs, '__len__'):
        assert inputs[0] is r[0]


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
def test_trt_docarray_inputs(make_trt_flow_fp16, inputs):
    c = Client(server=f'grpc://0.0.0.0:{make_trt_flow_fp16.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert isinstance(r, DocumentArray)
    assert r.embeddings.shape
    if hasattr(inputs, '__len__'):
        assert inputs[0] is r[0]
