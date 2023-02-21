import os

import pytest
from clip_server.model.clip import _transform_ndarray, _transform_blob
from clip_server.model.pretrained_models import download_model
from docarray import Document
from jina import Flow
import numpy as np


def test_server_download(tmpdir):
    download_model(
        url='https://clip-as-service.jina.ai/_static/favicon.png',
        target_folder=tmpdir,
        md5sum='66ea4817d73514888dcf6c7d2b00016d',
        with_resume=False,
    )
    target_path = os.path.join(tmpdir, 'favicon.png')
    file_size = os.path.getsize(target_path)
    assert file_size > 0

    part_path = target_path + '.part'
    with open(target_path, 'rb') as source, open(part_path, 'wb') as part_out:
        buf = source.read(10)
        part_out.write(buf)

    os.remove(target_path)

    download_model(
        url='https://clip-as-service.jina.ai/_static/favicon.png',
        target_folder=tmpdir,
        md5sum='66ea4817d73514888dcf6c7d2b00016d',
        with_resume=True,
    )
    assert os.path.getsize(target_path) == file_size
    assert not os.path.exists(part_path)


@pytest.mark.parametrize('md5', ['ABC', None, '43104e468ddd23c55bc662d84c87a7f8'])
def test_server_download_md5(tmpdir, md5):
    if md5 != 'ABC':
        download_model(
            url='https://clip-as-service.jina.ai/_static/favicon.png',
            target_folder=tmpdir,
            md5sum=md5,
            with_resume=False,
        )
    else:
        with pytest.raises(Exception):
            download_model(
                url='https://clip-as-service.jina.ai/_static/favicon.png',
                target_folder=tmpdir,
                md5sum=md5,
                with_resume=False,
            )


def test_server_download_not_regular_file(tmpdir):
    with pytest.raises(Exception):
        download_model(
            url='https://clip-as-service.jina.ai/_static/favicon.png',
            target_folder=tmpdir,
            md5sum='',
            with_resume=False,
        )
        download_model(
            url='https://docarray.jina.ai/_static/',
            target_folder=tmpdir,
            md5sum='',
            with_resume=False,
        )


def test_make_onnx_flow_wrong_name_path():
    from clip_server.executors.clip_onnx import CLIPEncoder

    with pytest.raises(Exception):
        encoder = CLIPEncoder(
            'ABC', model_path=os.path.expanduser('~/.cache/clip/ViT-B-32')
        )

    with pytest.raises(Exception) as info:
        encoder = CLIPEncoder('ViT-B/32', model_path='~/.cache/')


@pytest.mark.parametrize(
    'image_uri',
    [
        f'{os.path.dirname(os.path.abspath(__file__))}/img/00000.jpg',
        'https://clip-as-service.jina.ai/_static/favicon.png',
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
