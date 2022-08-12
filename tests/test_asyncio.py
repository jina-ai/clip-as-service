import asyncio
import os
import pytest

from clip_client import Client
from docarray import Document, DocumentArray


async def another_heavylifting_job():
    await asyncio.sleep(3)


@pytest.mark.asyncio
async def test_async_encode(make_flow):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    t1 = asyncio.create_task(another_heavylifting_job())
    t2 = asyncio.create_task(c.aencode(['hello world'] * 10))
    await asyncio.gather(t1, t2)
    assert t2.result().shape


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray([Document(text='hello, world'), Document(text='goodbye, world')]),
        DocumentArray(
            [
                Document(
                    uri='https://docarray.jina.ai/_static/favicon.png',
                    text='hello, world',
                ),
            ]
        ),
        DocumentArray.from_files(
            f'{os.path.dirname(os.path.abspath(__file__))}/**/*.jpg'
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_docarray_preserve_original_inputs(
    make_flow, inputs, port_generator
):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    t1 = asyncio.create_task(another_heavylifting_job())
    t2 = asyncio.create_task(c.aencode(inputs if not callable(inputs) else inputs()))
    await asyncio.gather(t1, t2)
    assert isinstance(t2.result(), DocumentArray)
    assert t2.result().embeddings.shape
    assert t2.result().contents == inputs.contents
    assert '__created_by_CAS__' not in t2.result()[0].tags
    assert '__loaded_by_CAS__' not in t2.result()[0].tags
    assert not t2.result()[0].tensor
    assert not t2.result()[0].blob
