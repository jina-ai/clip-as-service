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
                    uri='https://clip-as-service.jina.ai/_static/favicon.png',
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
async def test_async_docarray_preserve_original_inputs(make_flow, inputs):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    t1 = asyncio.create_task(another_heavylifting_job())
    t2 = asyncio.create_task(c.aencode(inputs if not callable(inputs) else inputs()))
    await asyncio.gather(t1, t2)
    assert isinstance(t2.result(), DocumentArray)
    assert inputs[0] is t2.result()[0]
    assert t2.result().embeddings.shape
    assert t2.result().contents == inputs.contents
    assert not t2.result()[0].tensor
    assert inputs[0] is t2.result()[0]


@pytest.mark.parametrize(
    'inputs',
    [
        [Document(id=str(i), text='hello, world') for i in range(20)],
        DocumentArray([Document(id=str(i), text='hello, world') for i in range(20)]),
    ],
)
@pytest.mark.asyncio
async def test_async_docarray_preserve_original_order(make_flow, inputs):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    t1 = asyncio.create_task(another_heavylifting_job())
    t2 = asyncio.create_task(
        c.aencode(inputs if not callable(inputs) else inputs(), batch_size=1)
    )
    await asyncio.gather(t1, t2)
    assert isinstance(t2.result(), DocumentArray)
    for i in range(len(inputs)):
        assert inputs[i] is t2.result()[i]
        assert inputs[i].id == str(i)
