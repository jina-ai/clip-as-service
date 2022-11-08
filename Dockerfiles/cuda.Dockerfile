ARG CUDA_VERSION=11.4.2

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ARG JINA_VERSION=3.11.0
ARG BACKEND_TAG=torch

# constant, wont invalidate cache
LABEL org.opencontainers.image.vendor="Jina AI Limited" \
      org.opencontainers.image.licenses="Apache 2.0" \
      org.opencontainers.image.title="CLIP-as-Service" \
      org.opencontainers.image.description="Embed images and sentences into fixed-length vectors with CLIP" \
      org.opencontainers.image.authors="hello@jina.ai" \
      org.opencontainers.image.url="clip-as-service" \
      org.opencontainers.image.documentation="https://clip-as-service.jina.ai/"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-setuptools python3-wheel python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*;

RUN python3 -m pip install --default-timeout=1000 --no-cache-dir torch torchvision torchaudio nvidia-pyindex transformers --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install --default-timeout=1000 --no-cache-dir "jina[standard]==${JINA_VERSION}"

# copy will almost always invalid the cache
COPY . /cas/

WORKDIR /cas

RUN if [ "${BACKEND_TAG}" != "torch" ]; then python3 -m pip install --no-cache-dir "./[${BACKEND_TAG}]" ;  \
    else python3 -m pip install --no-cache-dir "./[transformers]" ; fi && python3 -m pip install --no-cache-dir .

RUN echo "\
jtype: CLIPEncoder\n\
metas:\n\
  py_modules:\n\
    - clip_server.executors.clip_$BACKEND_TAG\n\
" > /tmp/config.yml

ENTRYPOINT ["jina", "executor", "--uses", "/tmp/config.yml"]





