ARG CUDA_VERSION=11.4.2

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ARG JINA_VERSION=3.3.25
ARG PIP_TAG

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-setuptools python3-wheel python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*;

RUN python3 -m pip install --default-timeout=1000 --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install --default-timeout=1000 --no-cache-dir "jina[standard]==${JINA_VERSION}"

RUN python3 -m pip install nvidia-pyindex

# copy will almost always invalid the cache
COPY . /clip-as-service/


RUN echo '\
jtype: CLIPEncoder\n\
with:\n\
  name: ${{ env.MODEL_NAME }}\n\
  device: ${{ env.DEVICE }}\n\
  minibatch_size: ${{ env.MINIBATCH_SIZE }}\n\
metas:\n\
  py_modules:\n\
    - server/clip_server/executors/clip_${{ env.ENGINE }}.py\n\
' > /tmp/config.yml

RUN cd /clip-as-service && \
    if [ -n "${PIP_TAG}" ]; then python3 -m pip install --no-cache-dir server/"[${PIP_TAG}]" ; fi && \
    python3 -m pip install --no-cache-dir "server/"

WORKDIR /clip-as-service

ENTRYPOINT ["jina", "executor", "--uses", "/tmp/config.yml"]





