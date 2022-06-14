# Dockerfile to run Clip-as-Service with TensorRT, CUDA integration

ARG TENSORRT_VERSION=22.04

FROM nvcr.io/nvidia/tensorrt:${TENSORRT_VERSION}-py3

ARG JINA_VERSION=3.6.0
ARG BACKEND_TAG=tensorrt

RUN pip3 install --default-timeout=1000 --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 -m pip install --default-timeout=1000 --no-cache-dir "jina[standard]==${JINA_VERSION}"

# copy will almost always invalid the cache
COPY . /clip_server/

RUN echo '\
jtype: CLIPEncoder\n\
metas:\n\
  py_modules:\n\
    - clip_server/executors/clip_$BACKEND_TAG.py\n\
' > /tmp/config.yml

RUN cd /clip_server/ && python3 -m pip install --no-cache-dir "./[$BACKEND_TAG]"

WORKDIR /clip_server

ENTRYPOINT ["jina", "executor", "--uses", "/tmp/config.yml"]

