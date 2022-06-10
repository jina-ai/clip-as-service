import time
import random
from jina import Executor, requests, DocumentArray


class Toy2(Executor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @requests(on='/')
    def encode(self, docs: DocumentArray, **kwargs):
        time.sleep(random.random())
        return docs


if __name__ == '__main__':
    Toy2.serve(port=51000)