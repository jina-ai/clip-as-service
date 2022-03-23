import asyncio

import pytest

from clip_client import Client


async def another_heavylifting_job():
    await asyncio.sleep(3)


@pytest.mark.asyncio
async def test_async_encode(make_flow):
    c = Client(server=f'grpc://0.0.0.0:{make_flow.port}')
    t1 = asyncio.create_task(another_heavylifting_job())
    t2 = asyncio.create_task(c.aencode(['hello world'] * 10))
    await asyncio.gather(t1, t2)
    assert t2.result().shape
