import time
import random
import numpy as np
from jina import Executor, requests, DocumentArray, Flow


class Toy2(Executor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @requests
    async def encode(self, docs: DocumentArray, **kwargs):
        for d in docs:
            d.embedding = np.random.rand(100)
        time.sleep(random.random()*3)
        return docs


if __name__ == '__main__':
    f = Flow(port=51000).add(uses=Toy2, replicas=4)
    with f:
        f.block()
