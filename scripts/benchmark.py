import random
import time
from typing import Optional
import threading
import click
import numpy as np
from docarray import Document, DocumentArray


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


np.random.seed(123)


class BenchmarkClient(threading.Thread):
    def __init__(
        self,
        server: str,
        batch_size: int = 1,
        modality: str = 'text',
        num_iter: Optional[int] = 100,
        image_sample: str = None,
        **kwargs,
    ):
        """
        @param server: the clip-as-service server URI
        @param batch_size: number of batch sample
        @param num_iter: number of repeat run per experiment
        @param image_sample: uri of the test image
        """
        assert num_iter > 2, 'num_iter must be greater than 2'
        super().__init__()
        self.server = server
        self.batch_size = batch_size
        self.modality = modality
        self.image_sample = image_sample
        self.num_iter = num_iter
        self.avg_time = 0

    def run(self):
        try:
            from clip_client import Client
        except ImportError:
            raise ImportError(
                'clip_client module is not available. it is required for benchmarking.'
                'Please use ""pip install clip-client" to install it.'
            )

        if self.modality == 'text':
            from clip_server.model.simple_tokenizer import SimpleTokenizer

            tokenizer = SimpleTokenizer()
            vocab = list(tokenizer.encoder.keys())
            batch = DocumentArray(
                [
                    Document(text=' '.join(random.choices(vocab, k=78)))
                    for _ in range(self.batch_size)
                ]
            )
        elif self.modality == 'image':
            batch = DocumentArray(
                [
                    Document(blob=open(self.image_sample, 'rb').read())
                    for _ in range(self.batch_size)
                ]
            )
        else:
            raise ValueError(f'The modality "{self.modality}" is unsupported')

        client = Client(self.server)

        time_costs = []
        for _ in range(self.num_iter):
            start = time.perf_counter()
            r = client.encode(batch, batch_size=self.batch_size)
            time_costs.append(time.perf_counter() - start)
        self.avg_time = np.mean(time_costs[2:])


@click.command(name='clip-as-service benchmark')
@click.argument('server')
@click.option(
    '--batch_sizes',
    multiple=True,
    type=int,
    default=[1, 8, 16, 32, 64],
    help='number of batch',
)
@click.option(
    '--num_iter', default=10, help='number of repeat run per experiment (must > 2)'
)
@click.option(
    "--concurrent_clients",
    multiple=True,
    type=int,
    default=[1, 4, 16, 32, 64],
    help='number of concurrent clients per experiment',
)
@click.option("--image_sample", help='path to the image sample file')
def main(server, batch_sizes, num_iter, concurrent_clients, image_sample):
    # wait until the server is ready
    for batch_size in batch_sizes:
        for num_client in concurrent_clients:
            all_clients = [
                BenchmarkClient(
                    server,
                    batch_size=batch_size,
                    num_iter=num_iter,
                    modality='image' if (image_sample is not None) else 'text',
                    image_sample=image_sample,
                )
                for _ in range(num_client)
            ]

            for bc in all_clients:
                bc.start()

            clients_speed = []
            for bc in all_clients:
                bc.join()
                clients_speed.append(batch_size / bc.avg_time)

            max_speed, min_speed, avg_speed = (
                max(clients_speed),
                min(clients_speed),
                np.mean(clients_speed),
            )

            print(
                '(concurrent client=%d, batch_size=%d) avg speed: %.3f\tmax speed: %.3f\tmin speed: %.3f'
                % (num_client, batch_size, avg_speed, max_speed, min_speed),
                flush=True,
            )


if __name__ == '__main__':
    main()
