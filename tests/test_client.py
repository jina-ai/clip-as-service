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


class ErrorExec(Executor):
    @requests
    def foo(self, docs, **kwargs):
        raise NotImplementedError


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
        DocumentArray(),
    ],
)
@pytest.mark.parametrize('endpoint', ['encode', 'rank', 'index', 'search'])
@pytest.mark.asyncio
def test_empty_input(make_torch_flow, inputs, endpoint):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    r = getattr(c, endpoint)(inputs if not callable(inputs) else inputs())
    if endpoint == 'encode':
        if isinstance(inputs, DocumentArray):
            assert isinstance(r, DocumentArray)
        else:
            assert isinstance(r, list)
    else:
        assert isinstance(r, DocumentArray)
    assert len(r) == 0


@pytest.mark.parametrize(
    'inputs',
    [
        [],
        DocumentArray(),
    ],
)
@pytest.mark.parametrize('endpoint', ['aencode', 'arank', 'aindex', 'asearch'])
@pytest.mark.asyncio
async def test_async_empty_input(make_torch_flow, inputs, endpoint):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    r = await getattr(c, endpoint)(inputs if not callable(inputs) else inputs())
    if endpoint == 'aencode':
        if isinstance(inputs, DocumentArray):
            assert isinstance(r, DocumentArray)
        else:
            assert isinstance(r, list)
    else:
        assert isinstance(r, DocumentArray)
    assert len(r) == 0


@pytest.mark.parametrize('endpoint', ['encode', 'rank', 'index', 'search'])
def test_wrong_input_type(make_torch_flow, endpoint):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    with pytest.raises(Exception):
        getattr(c, endpoint)('hello')


@pytest.mark.parametrize('endpoint', ['aencode', 'arank', 'aindex', 'asearch'])
@pytest.mark.asyncio
async def test_wrong_input_type(make_torch_flow, endpoint):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    with pytest.raises(Exception):
        await getattr(c, endpoint)('hello')


@pytest.mark.parametrize('endpoint', ['encode', 'rank', 'index', 'search'])
@pytest.mark.slow
def test_custom_on_done(make_torch_flow, mocker, endpoint):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    on_done_mock = mocker.Mock()
    on_error_mock = mocker.Mock()
    on_always_mock = mocker.Mock()

    r = getattr(c, endpoint)(
        DocumentArray(
            [Document(text='hello', matches=DocumentArray([Document(text='jina')]))]
        ),
        on_done=on_done_mock,
        on_error=on_error_mock,
        on_always=on_always_mock,
    )
    assert r is None
    on_done_mock.assert_called_once()
    on_error_mock.assert_not_called()
    on_always_mock.assert_called_once()


@pytest.mark.parametrize('endpoint', ['aencode', 'arank', 'aindex', 'asearch'])
@pytest.mark.slow
@pytest.mark.asyncio
async def test_async_custom_on_done(make_torch_flow, mocker, endpoint):
    from clip_client.client import Client

    c = Client(server=f'grpc://0.0.0.0:{make_torch_flow.port}')

    on_done_mock = mocker.Mock()
    on_error_mock = mocker.Mock()
    on_always_mock = mocker.Mock()

    r = await getattr(c, endpoint)(
        DocumentArray(
            [Document(text='hello', matches=DocumentArray([Document(text='jina')]))]
        ),
        on_done=on_done_mock,
        on_error=on_error_mock,
        on_always=on_always_mock,
    )
    assert r is None
    on_done_mock.assert_called_once()
    on_error_mock.assert_not_called()
    on_always_mock.assert_called_once()


@pytest.mark.parametrize('endpoint', ['encode', 'rank', 'index', 'search'])
@pytest.mark.slow
def test_custom_on_error(port_generator, mocker, endpoint):
    from clip_client.client import Client

    f = Flow(port=port_generator()).add(uses=ErrorExec)

    with f:
        c = Client(server=f'grpc://0.0.0.0:{f.port}')

        on_done_mock = mocker.Mock()
        on_error_mock = mocker.Mock()
        on_always_mock = mocker.Mock()

        r = getattr(c, endpoint)(
            DocumentArray(
                [Document(text='hello', matches=DocumentArray([Document(text='jina')]))]
            ),
            on_done=on_done_mock,
            on_error=on_error_mock,
            on_always=on_always_mock,
        )
        assert r is None
        on_done_mock.assert_not_called()
        on_error_mock.assert_called_once()
        on_always_mock.assert_called_once()


@pytest.mark.parametrize('endpoint', ['aencode', 'arank', 'aindex', 'asearch'])
@pytest.mark.slow
@pytest.mark.asyncio
async def test_async_custom_on_error(port_generator, mocker, endpoint):
    from clip_client.client import Client

    f = Flow(port=port_generator()).add(uses=ErrorExec)

    with f:
        c = Client(server=f'grpc://0.0.0.0:{f.port}')

        on_done_mock = mocker.Mock()
        on_error_mock = mocker.Mock()
        on_always_mock = mocker.Mock()

        r = await getattr(c, endpoint)(
            DocumentArray(
                [Document(text='hello', matches=DocumentArray([Document(text='jina')]))]
            ),
            on_done=on_done_mock,
            on_error=on_error_mock,
            on_always=on_always_mock,
        )
        assert r is None
        on_done_mock.assert_not_called()
        on_error_mock.assert_called_once()
        on_always_mock.assert_called_once()
