import pytest
import numpy as np
from clip_server.executors.helper import numpy_softmax


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
