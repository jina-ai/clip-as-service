import clip_client
import jina
from jina import Executor, requests, DocumentArray, Document
import uuid
import sys

if __name__ == '__main__':
    tag = sys.argv[1]

    da = DocumentArray([Document(id=f'{tag}-{i}', text='hello') for i in range(10)])

    print(f'request inputs: {[d.id for d in da]}')

    client = jina.clients.Client(port=51001)
    result = client.post(on='/', inputs=da, request_size=1024)

    print(f'response results: {[d.id for d in result]}')
    assert all([d.id.startswith(f'{tag}') for d in result])
