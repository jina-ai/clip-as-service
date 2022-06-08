import os

import pytest
import numpy as np
from docarray import Document, DocumentArray
from jina import Flow

from clip_client.client import Client


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(tensor=np.ones((3, 224, 224), dtype=np.float32)),
                Document(tensor=np.ones((3, 224, 224), dtype=np.float32)),
            ]
        ),
    ],
)
def test_batch_no_preprocessing(make_hg_flow_no_default, inputs, port_generator):
    c = Client(server=f'grpc://0.0.0.0:{make_hg_flow_no_default.port}')
    r = c.encode(inputs if not callable(inputs) else inputs())
    assert len(r) == 2
    assert r[0].shape == (512,)
    assert r[0].dtype == np.float32
