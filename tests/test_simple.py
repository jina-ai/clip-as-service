import os

import pytest
from docarray import Document, DocumentArray
from jina import Flow

from clip_client.client import Client


@pytest.mark.parametrize('protocol', ['grpc', 'http', 'websocket', 'other'])
@pytest.mark.parametrize('jit', [True, False])
def test_protocols(port_generator, protocol, jit, pytestconfig):
    from clip_server.executors.clip_torch import CLIPEncoder

    if protocol == 'other':
        with pytest.raises(ValueError):
            Client(server=f'{protocol}://0.0.0.0:8000')
        return

    f = Flow(port=port_generator(), protocol=protocol).add(
        uses=CLIPEncoder, uses_with={'jit': jit}
    )
    with f:
        c = Client(server=f'{protocol}://0.0.0.0:{f.port}')
        c.profile()
        c.profile(content='hello world')
        c.profile(content=f'{pytestconfig.rootdir}/tests/img/00000.jpg')


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
def test_plain_inputs(make_flow, inputs, port_generator):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert (
        r.shape[0] == len(list(inputs)) if not callable(inputs) else len(list(inputs()))
    )


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
def test_docarray_inputs(make_flow, inputs, port_generator):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert isinstance(r, DocumentArray)
    assert r.embeddings.shape
    assert '__created_by_CAS__' not in r[0].tags
    assert '__loaded_by_CAS__' not in r[0].tags
    assert not r[0].tensor
    assert not r[0].blob
    if hasattr(inputs, '__len__'):
        assert inputs[0] is r[0]


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
def test_docarray_preserve_original_inputs(make_flow, inputs, port_generator):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert isinstance(r, DocumentArray)
    assert r.embeddings.shape
    assert r.contents == inputs.contents
    assert '__created_by_CAS__' not in r[0].tags
    assert '__loaded_by_CAS__' not in r[0].tags
    assert not r[0].tensor
    assert not r[0].blob
    assert inputs[0] is r[0]


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
def test_docarray_traversal(make_flow, inputs, port_generator):
    from jina import Client as _Client

    da = DocumentArray.empty(1)
    da[0].chunks = inputs

    c = _Client(host=f'grpc://0.0.0.0', port=make_flow.port)
    r1 = c.post(on='/', inputs=da, parameters={'traversal_paths': '@c'})
    assert isinstance(r1, DocumentArray)
    assert r1[0].chunks.embeddings.shape[0] == len(inputs)
    assert '__created_by_CAS__' not in r1[0].tags
    assert '__loaded_by_CAS__' not in r1[0].tags
    assert not r1[0].tensor
    assert not r1[0].blob
    assert not r1[0].chunks[0].tensor
    assert not r1[0].chunks[0].blob

    r2 = c.post(on='/', inputs=da, parameters={'access_paths': '@c'})
    assert isinstance(r2, DocumentArray)
    assert r2[0].chunks.embeddings.shape[0] == len(inputs)
    assert '__created_by_CAS__' not in r2[0].tags
    assert '__loaded_by_CAS__' not in r2[0].tags
    assert not r2[0].tensor
    assert not r2[0].blob
    assert not r2[0].chunks[0].tensor
    assert not r2[0].chunks[0].blob
