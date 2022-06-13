import clip_client
import jina
from jina import Executor, requests, DocumentArray, Document, Flow
import uuid
import sys


class Toy1(Executor):
    def __init__(self, local_server: str, **kwargs):
        super().__init__(**kwargs)
        self._client = clip_client.Client(server=local_server)

    @requests(on='/')
    async def do_something(self, docs: DocumentArray, **kwargs):

        results = await self._client.aencode(docs, request_size=1024)
        return results


if __name__ == '__main__':
    f = Flow(port=51001).add(
        uses=Toy1, uses_with={'local_server': 'grpc://0.0.0.0:51000'}
    )
    with f:
        f.block()
