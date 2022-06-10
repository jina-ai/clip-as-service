import clip_client
import jina
from jina import Executor, requests, DocumentArray, Document
import uuid
import sys


def do_something(docs):
    server_url = 'grpc://0.0.0.0:51001'

    da = docs.post(server_url)

    print(f'before: {[d.id for d in docs]} +++ {[d.matches[0].text for d in docs]}')
    print(f'after: {[d.id for d in da]} +++ {[d.matches[0].text for d in da]}')
    return da


if __name__ == '__main__':
    tag = sys.argv[1]
    # uri = 'https://raw.githubusercontent.com/jina-ai/clip-as-service/main/.github/README-img/Hurst-began-again.png'
    da = DocumentArray()
    for _ in range(20):
        da.append(
            Document(
                id=f'{tag}-{_}', text='hello', matches=[Document(text=f'{tag}+{_}')]
            )
        )
    do_something(da)
