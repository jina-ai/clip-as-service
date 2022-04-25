import os

import pytest
from clip_client import Client
from clip_server.executors.clip_torch import CLIPEncoder
from docarray import DocumentArray, Document
from jina import Flow


@pytest.mark.asyncio
async def test_torch_executor_rank_img2texts():
    ce = CLIPEncoder()

    da = DocumentArray.from_files(
        f'{os.path.dirname(os.path.abspath(__file__))}/**/*.jpg'
    )
    for d in da:
        d.matches.append(Document(text='hello, world!'))
        d.matches.append(Document(text='goodbye, world!'))

    await ce.rerank(da, {})
    print(da['@m', 'scores__clip-rank__value'])
    for d in da:
        for c in d.matches:
            assert c.scores['clip-rank'].value is not None


@pytest.mark.asyncio
async def test_torch_executor_rank_text2imgs():
    ce = CLIPEncoder()
    db = DocumentArray(
        [Document(text='hello, world!'), Document(text='goodbye, world!')]
    )
    for d in db:
        d.matches.extend(
            DocumentArray.from_files(
                f'{os.path.dirname(os.path.abspath(__file__))}/**/*.jpg'
            )
        )
    await ce.rerank(db, {})
    print(db['@m', 'scores__clip-rank__value'])
    for d in db:
        for c in d.matches:
            assert c.scores['clip-rank'].value is not None


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
def test_docarray_inputs(d, port_generator):
    from clip_server.executors.clip_torch import CLIPEncoder

    f = Flow(port=port_generator()).add(uses=CLIPEncoder)
    with f:
        c = Client(server=f'grpc://0.0.0.0:{f.port}')
        r = c.rerank([d])
    assert isinstance(r, DocumentArray)
    rv = r['@m', 'scores__clip-rank__value']
    for v in rv:
        assert v is not None
        assert v > 0
