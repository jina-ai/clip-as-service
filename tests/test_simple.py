import os

import pytest
from docarray import Document, DocumentArray
from jina import Flow

from clip_client.client import Client


@pytest.mark.parametrize('protocol', ['grpc', 'http', 'websocket', 'other'])
def test_protocols(port_generator, protocol, pytestconfig):
    from clip_server.executors.clip_torch import CLIPEncoder

    if protocol == 'other':
        with pytest.raises(ValueError):
            Client(server=f'{protocol}://0.0.0.0:8000')
        return

    f = Flow(port=port_generator(), protocol=protocol).add(uses=CLIPEncoder)
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
    assert r.shape


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
