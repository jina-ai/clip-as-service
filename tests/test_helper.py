import pytest
import numpy as np
from clip_server.executors.helper import numpy_softmax
from clip_server.executors.helper import split_img_txt_da
from clip_server.executors.helper import preproc_image
from docarray import Document, DocumentArray


@pytest.mark.parametrize('shape', [(5, 10), (5, 10, 10)])
@pytest.mark.parametrize('axis', [-1, 1, 0])
def test_numpy_softmax(shape, axis):
    import torch

    logits = np.random.random(shape)

    np_softmax = numpy_softmax(logits, axis=axis)
    torch_softmax = torch.from_numpy(logits).softmax(dim=axis).numpy()
    np.testing.assert_array_almost_equal(np_softmax, torch_softmax)

    np_softmax = numpy_softmax(logits, axis=axis)
    torch_softmax = torch.from_numpy(logits).softmax(dim=axis).numpy()
    np.testing.assert_array_almost_equal(np_softmax, torch_softmax)


@pytest.mark.parametrize(
    'inputs',
    [
        (
            DocumentArray(
                [
                    Document(text='hello, world'),
                    Document(text='goodbye, world'),
                    Document(
                        text='hello, world',
                        uri='https://docarray.jina.ai/_static/favicon.png',
                    ),
                    Document(
                        uri='https://docarray.jina.ai/_static/favicon.png',
                    ),
                ]
            ),
            (3, 1),
        ),
        (
            DocumentArray(
                [
                    Document(text='hello, world'),
                    Document(tensor=np.array([0, 1, 2])),
                    Document(
                        uri='https://docarray.jina.ai/_static/favicon.png'
                    ).load_uri_to_blob(),
                    Document(
                        tensor=np.array([0, 1, 2]),
                        uri='https://docarray.jina.ai/_static/favicon.png',
                    ),
                    Document(
                        uri='https://docarray.jina.ai/_static/favicon.png',
                    ),
                ]
            ),
            (1, 4),
        ),
        (
            DocumentArray(
                [
                    Document(text='hello, world'),
                    Document(uri='https://docarray.jina.ai/_static/favicon.png'),
                ]
            ),
            (1, 1),
        ),
    ],
)
def test_split_img_txt_da(inputs):
    txt_da = DocumentArray()
    img_da = DocumentArray()
    for doc in inputs[0]:
        split_img_txt_da(doc, img_da, txt_da)
    assert len(txt_da) == inputs[1][0]
    assert len(img_da) == inputs[1][1]


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    uri='https://docarray.jina.ai/_static/favicon.png',
                    tags={'__loaded_by_CAS__': True},
                ).load_uri_to_blob(),
                Document(
                    uri='https://docarray.jina.ai/_static/favicon.png',
                ).load_uri_to_blob(),
            ]
        )
    ],
)
def test_preproc_image(inputs):
    from clip_server.model import clip

    preprocess_fn = clip._transform_ndarray(224)
    da, pixel_values = preproc_image(inputs, preprocess_fn)
    assert len(da) == 2
    assert not da[0].blob
    assert da[1].blob
    assert not da[0].tensor
    assert not da[1].tensor
    assert pixel_values.get('pixel_values') is not None
