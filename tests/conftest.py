import pytest
from jina import helper, Flow


@pytest.fixture(scope='session')
def port_generator():
    generated_ports = set()

    def random_port():
        port = helper.random_port()
        while port in generated_ports:
            port = helper.random_port()
        generated_ports.add(port)
        return port

    return random_port


@pytest.fixture(scope='session', params=['onnx', 'torch', 'hg', 'onnx_custom'])
def make_flow(port_generator, request):
    if request.param != 'onnx_custom':
        if request.param == 'onnx':
            from clip_server.executors.clip_onnx import CLIPEncoder
        elif request.param == 'torch':
            from clip_server.executors.clip_torch import CLIPEncoder
        else:
            from clip_server.executors.clip_hg import CLIPEncoder

        f = Flow(port=port_generator()).add(name=request.param, uses=CLIPEncoder)
    else:
        import os
        from clip_server.executors.clip_onnx import CLIPEncoder

        f = Flow(port=port_generator()).add(
            name=request.param,
            uses=CLIPEncoder,
            uses_with={'model_path': os.path.expanduser('~/.cache/clip/v2/ViT-B-32')},
        )
    with f:
        yield f


@pytest.fixture(scope='session', params=['torch'])
def make_torch_flow(port_generator, request):
    from clip_server.executors.clip_torch import CLIPEncoder

    f = Flow(port=port_generator()).add(name=request.param, uses=CLIPEncoder)
    with f:
        yield f


@pytest.fixture(scope='session', params=['tensorrt'])
def make_trt_flow(port_generator, request):
    from clip_server.executors.clip_tensorrt import CLIPEncoder

    f = Flow(port=port_generator()).add(name=request.param, uses=CLIPEncoder)
    with f:
        yield f


@pytest.fixture(scope='session', params=['hg'])
def make_hg_flow(port_generator, request):
    from clip_server.executors.clip_hg import CLIPEncoder

    f = Flow(port=port_generator()).add(name=request.param, uses=CLIPEncoder)
    with f:
        yield f


@pytest.fixture(scope='session', params=['hg'])
def make_hg_flow_no_default(port_generator, request):
    from clip_server.executors.clip_hg import CLIPEncoder

    f = Flow(port=port_generator()).add(
        name=request.param,
        uses=CLIPEncoder,
        uses_with={'preprocessing': False},
    )
    with f:
        yield f
