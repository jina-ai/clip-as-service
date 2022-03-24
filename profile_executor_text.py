from time import perf_counter

from docarray import DocumentArray, Document

from clip_server.executors.clip_torch import CLIPEncoder
# from clip_server.executors.clip_torch_dataloader import CLIPEncoder

executor = CLIPEncoder()


da = DocumentArray([Document(text='random text here') for _ in range(30000)])

async def main():
    await executor.encode(da)

import asyncio
st = perf_counter()
asyncio.run(main())

print('first time', perf_counter()-st)


st = perf_counter()
asyncio.run(main())

print('second time', perf_counter()-st)
