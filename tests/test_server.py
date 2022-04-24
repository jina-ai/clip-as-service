import os

import pytest
from clip_server.model.clip import _transform_ndarray, _transform_blob, _download
from docarray import Document
import numpy as np


def test_server_download(tmpdir):
    _download('https://docarray.jina.ai/_static/favicon.png', tmpdir, with_resume=False)

    target_path = os.path.join(tmpdir, 'favicon.png')
    file_size = os.path.getsize(target_path)
    assert file_size > 0

    part_path = target_path + '.part'
    with open(target_path, 'rb') as source, open(part_path, 'wb') as part_out:
        buf = source.read(10)
        part_out.write(buf)

    os.remove(target_path)

    _download('https://docarray.jina.ai/_static/favicon.png', tmpdir, with_resume=True)
    assert os.path.getsize(target_path) == file_size
    assert not os.path.exists(part_path)


@pytest.mark.parametrize(
    'image_uri',
    [
        f'{os.path.dirname(os.path.abspath(__file__))}/img/00000.jpg',
        'https://docarray.jina.ai/_static/favicon.png',
    ],
)
@pytest.mark.parametrize('size', [224, 288, 384, 448])
def test_server_preprocess_ndarray_image(image_uri, size):
    d1 = Document(uri=image_uri)
    d1.load_uri_to_blob()
    d2 = Document(uri=image_uri)
    d2.load_uri_to_image_tensor()

    t1 = _transform_blob(size)(d1.blob).numpy()
    t2 = _transform_ndarray(size)(d2.tensor).numpy()
    assert t1.shape == t2.shape


@pytest.mark.parametrize(
    'tensor',
    [
        np.random.random([100, 100, 3]),
        np.random.random([1, 1, 3]),
        np.random.random([5, 50, 3]),
    ],
)
def test_transform_arbitrary_tensor(tensor):
    d = Document(tensor=tensor)
    assert _transform_ndarray(224)(d.tensor).numpy().shape == (3, 224, 224)
