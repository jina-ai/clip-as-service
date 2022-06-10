import clip_client
import jina
from jina import Executor, requests, DocumentArray, Document
import uuid
import sys

class Toy1(Executor):

    def __init__(self, local_server: str, **kwargs):
        super().__init__(**kwargs)
        # self._client = jina.clients.Client(host=local_server, asyncio=True)
        self._client = clip_client.Client(server=local_server)

    @requests(on='/')
    async def do_something(self, docs: DocumentArray, **kwargs):
        print(f'==before==: {len(docs)}, matches count: {len(docs[0].matches)}', docs[0].matches[:, ['id', 'text']])
        print('*' * 30)


        # results = [i async for i in self._client.post(on='/encode', inputs=docs, request_size=10)][0]
        results = await self._client.aencode(docs)

        print('==after==')
        print(f'docs count: {len(results)}, matches count: {len(results["@m"])}', results['@m', ['id', 'text']])
        print('-' * 30)
        print(f'first doc\'s matches count: {len(results[0].matches)}', results[0].matches[:, ('id', 'text')])
        print('*' * 30)

        return results
