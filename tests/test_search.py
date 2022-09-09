import os

import numpy as np
import pytest
from docarray import DocumentArray, Document

from clip_client import Client


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
@pytest.mark.parametrize('limit', [1, 2])
def test_index_search(make_search_flow, inputs, limit):
    c = Client(server=f'grpc://0.0.0.0:{make_search_flow.port}')

    r = c.index(inputs if not callable(inputs) else inputs())
    assert isinstance(r, DocumentArray)
    assert r.embeddings.shape[1] == 512

    r = c.search(inputs if not callable(inputs) else inputs(), limit=limit)
    assert isinstance(r, DocumentArray)
    for d in r:
        assert len(d.matches) == limit


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
@pytest.mark.parametrize('limit', [1, 2])
@pytest.mark.asyncio
async def test_async_index_search(make_search_flow, inputs, limit):
    c = Client(server=f'grpc://0.0.0.0:{make_search_flow.port}')
    r = await c.aindex(inputs if not callable(inputs) else inputs())
    assert isinstance(r, DocumentArray)

    assert r.embeddings.shape[1] == 512

    r = await c.asearch(inputs if not callable(inputs) else inputs(), limit=limit)
    assert isinstance(r, DocumentArray)
    for d in r:
        assert len(d.matches) == limit
