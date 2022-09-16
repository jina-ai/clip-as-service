import os
import random
import time
import pytest
import numpy as np
from docarray import Document, DocumentArray
from jina import Flow, Executor, requests


class Exec1(Executor):
    @requests
    async def aencode(self, docs, **kwargs):
        time.sleep(random.random() * 1)
        docs.embeddings = np.random.rand(len(docs), 10)


class Exec2(Executor):
    def __init__(self, server_host: str = '', **kwargs):
        super().__init__(**kwargs)
        from clip_client.client import Client

        self._client = Client(server=server_host)

    @requests
    async def process(self, docs, **kwargs):
        results = await self._client.aencode(docs, batch_size=2)
        return results


def test_client_concurrent_requests(port_generator):

    f1 = Flow(port=port_generator()).add(uses=Exec1)

    f2 = Flow(protocol='http').add(
        uses=Exec2, uses_with={'server_host': f'grpc://0.0.0.0:{f1.port}'}
    )

    with f1, f2:
        import jina
        from multiprocessing.pool import ThreadPool

        def run_post(docs):
            c = jina.clients.Client(port=f2.port, protocol='http')
            results = c.post(on='/', inputs=docs, request_size=2)
            # assert set([d.id for d in results]) != set([d.id for d in docs])
            return results

        def generate_docs(tag):
            return DocumentArray(
                [Document(id=f'{tag}_{i}', text='hello') for i in range(20)]
            )

        with ThreadPool(5) as p:
            results = p.map(run_post, [generate_docs(f't{k}') for k in range(5)])

        for r in results:
            assert len(set([d.id[:2] for d in r])) == 1


def test_client_large_input(make_torch_flow):
    from clip_client.client import Client

    inputs = ['hello' for _ in range(600)]

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')
    with pytest.warns(UserWarning):
        c.encode(inputs if not callable(inputs) else inputs())


@pytest.mark.parametrize(
    'inputs',
    [
        [],
        DocumentArray([]),
    ],
)
@pytest.mark.asyncio
async def test_client_empty_input(make_torch_flow, inputs):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    r = c.encode(inputs if not callable(inputs) else inputs())
    if isinstance(inputs, DocumentArray):
        assert isinstance(r, DocumentArray)
    else:
        assert isinstance(r, list)
    assert len(r) == 0

    r = await c.aencode(inputs if not callable(inputs) else inputs())
    if isinstance(inputs, DocumentArray):
        assert isinstance(r, DocumentArray)
    else:
        assert isinstance(r, list)
    assert len(r) == 0

    r = c.rank(inputs if not callable(inputs) else inputs())
    if isinstance(inputs, DocumentArray):
        assert isinstance(r, DocumentArray)
    else:
        assert isinstance(r, list)
    assert len(r) == 0

    r = await c.arank(inputs if not callable(inputs) else inputs())
    if isinstance(inputs, DocumentArray):
        assert isinstance(r, DocumentArray)
    else:
        assert isinstance(r, list)
    assert len(r) == 0


@pytest.mark.asyncio
async def test_str_input(make_torch_flow):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    with pytest.raises(Exception):
        c.encode('hello')
    with pytest.raises(Exception):
        await c.aencode('hello')
    with pytest.raises(Exception):
        c.rank('hello')
    with pytest.raises(Exception):
        await c.arank('hello')
    with pytest.raises(Exception):
        c.index('hello')
    with pytest.raises(Exception):
        await c.aindex('hello')
    with pytest.raises(Exception):
        c.search('hello')
    with pytest.raises(Exception):
        await c.asearch('hello')
