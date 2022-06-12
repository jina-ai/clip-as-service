import clip_client
import jina
from jina import Executor, requests, DocumentArray, Document, Flow
import uuid
import sys


class Toy1(Executor):
    def __init__(self, local_server: str, **kwargs):
        super().__init__(**kwargs)
        self._client = jina.clients.Client(host=local_server, asyncio=True)
        # self._client = clip_client.Client(server=local_server)

    @requests(on='/')
    async def do_something(self, docs: DocumentArray, **kwargs):
        results = [i async for i in self._client.post(on='/encode', inputs=docs, request_size=2)][0]
        # results = await self._client.aencode(docs)

        # results vs docs
        print(f'before: {[d.id for d in docs]}')
        print(f'after: {[d.id for d in results]}')
        return results


if __name__ == '__main__':
    f = Flow(port=51001).add(uses=Toy1, uses_with={'local_server': 'grpc://0.0.0.0:51000'})
    with f:
        f.block()
