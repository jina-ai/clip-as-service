import os

import pytest
from clip_client import Client
from clip_server.executors.clip_torch import CLIPEncoder as TorchCLIPEncoder
from clip_server.executors.clip_onnx import CLIPEncoder as ONNXCLILPEncoder
from docarray import DocumentArray, Document


@pytest.mark.asyncio
@pytest.mark.parametrize('encoder_class', [TorchCLIPEncoder, ONNXCLILPEncoder])
async def test_torch_executor_rank_img2texts(encoder_class):
    ce = encoder_class()

    da = DocumentArray.from_files(
        f'{os.path.dirname(os.path.abspath(__file__))}/**/*.jpg'
    )
    for d in da:
        d.matches.append(Document(text='hello, world!'))
        d.matches.append(Document(text='goodbye, world!'))

    await ce.rank(da, {})
    print(da['@m', 'scores__clip_score__value'])
    for d in da:
        for c in d.matches:
            assert c.scores['clip_score'].value is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('encoder_class', [TorchCLIPEncoder, ONNXCLILPEncoder])
async def test_torch_executor_rank_text2imgs(encoder_class):
    ce = encoder_class()
    db = DocumentArray(
        [Document(text='hello, world!'), Document(text='goodbye, world!')]
    )
    for d in db:
        d.matches.extend(
            DocumentArray.from_files(
                f'{os.path.dirname(os.path.abspath(__file__))}/**/*.jpg'
            )
        )
    await ce.rank(db, {})
    print(db['@m', 'scores__clip_score__value'])
    for d in db:
        for c in d.matches:
            assert c.scores['clip_score'].value is not None


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
def test_docarray_inputs(make_flow, d):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    r = c.rank([d])
    assert isinstance(r, DocumentArray)
    rv = r['@m', 'scores__clip_score__value']
    for v in rv:
        assert v is not None
        assert v > 0


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
@pytest.mark.asyncio
async def test_async_arank(make_flow, d):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    r = await c.arank([d])
    assert isinstance(r, DocumentArray)
    rv = r['@m', 'scores__clip_score__value']
    for v in rv:
        assert v is not None
        assert v > 0

    rv = r['@m', 'scores__clip_score_cosine__value']
    for v in rv:
        assert v is not None
        assert v > 0
