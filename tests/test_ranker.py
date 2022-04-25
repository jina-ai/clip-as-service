import os

import pytest
from clip_server.executors.clip_torch import CLIPEncoder
from docarray import DocumentArray, Document


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
