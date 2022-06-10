import time
import random
import numpy as np
from jina import Executor, requests, DocumentArray, Flow


class Toy2(Executor):
    @requests
    async def encode(self, docs: DocumentArray, **kwargs):
        docs.embeddings = np.random.rand(len(docs), 100)
        time.sleep(random.random() * 3)

        return docs


if __name__ == '__main__':
    f = Flow(port=51000).add(uses=Toy2, replicas=4)
    with f:
        f.block()
