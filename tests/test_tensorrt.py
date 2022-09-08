import os

import pytest
import numpy as np
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
    assert inputs[0] is r[0]
    assert r.embeddings.shape
    if hasattr(inputs, '__len__'):
        assert inputs[0] is r[0]


@pytest.mark.gpu
@pytest.mark.asyncio
@pytest.mark.parametrize(
    'd',
    [
        Document(
            uri='https://docarray.jina.ai/_static/favicon.png',
            matches=[Document(text='hello, world'), Document(text='goodbye, world')],
        ),
        Document(
            text='hello, world',
            matches=[
                Document(uri='https://docarray.jina.ai/_static/favicon.png'),
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/img/00000.jpg'
                ),
            ],
        ),
    ],
)
async def test_async_arank(make_trt_flow, d):
    c = Client(server=f'grpc://0.0.0.0:{make_trt_flow.port}')
    r = await c.arank([d])
    assert isinstance(r, DocumentArray)
    assert d is r[0]
    rv = r['@m', 'scores__clip_score__value']
    for v in rv:
        assert v is not None
        assert v > 0
    np.testing.assert_almost_equal(sum(rv), 1.0)

    rv = r['@m', 'scores__clip_score_cosine__value']
    for v in rv:
        assert v is not None
        assert -1.0 <= v <= 1.0
