# Dockerfile to run Clip-as-Service with TensorRT, CUDA integration

ARG TENSORRT_VERSION=22.04

FROM nvcr.io/nvidia/tensorrt:${TENSORRT_VERSION}-py3

ARG JINA_VERSION=3.6.0
ARG BACKEND_TAG=tensorrt

# constant, wont invalidate cache
LABEL org.opencontainers.image.vendor="Jina AI Limited" \
      org.opencontainers.image.licenses="Apache 2.0" \
      org.opencontainers.image.title="CLIP-as-Service" \
      org.opencontainers.image.description="Embed images and sentences into fixed-length vectors with CLIP" \
      org.opencontainers.image.authors="hello@jina.ai" \
      org.opencontainers.image.url="clip-as-service" \
      org.opencontainers.image.documentation="https://clip-as-service.jina.ai/"

RUN pip3 install --default-timeout=1000 --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 -m pip install --default-timeout=1000 --no-cache-dir "jina[standard]==${JINA_VERSION}"

# copy will almost always invalid the cache
COPY . /cas/
WORKDIR /cas

RUN python3 -m pip install --no-cache-dir "./[$BACKEND_TAG]"

RUN CLIP_PATH=$(python -c "import clip_server;print(clip_server.__path__[0])") \
    && echo '\
jtype: CLIPEncoder\n\
metas:\n\
  py_modules:\n\
    - $CLIP_PATH/executors/clip_$BACKEND_TAG.py\n\
' > /tmp/config.yml



ENTRYPOINT ["jina", "executor", "--uses", "/tmp/config.yml"]

