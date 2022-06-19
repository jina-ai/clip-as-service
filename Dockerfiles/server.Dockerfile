ARG CUDA_VERSION=11.6.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

WORKDIR /cas

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip wget \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.4.0.27-1+cuda11.6_amd64.deb \
    && apt install ./libcudnn8_8.4.0.27-1+cuda11.6_amd64.deb  \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

COPY server ./server
RUN pip install ./server/
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=cas
ARG GROUP_NAME=cas

RUN groupadd -g ${GROUP_ID} ${USER_NAME} &&\
    useradd -l -u ${USER_ID} -g ${USER_NAME} ${GROUP_NAME} &&\
    mkdir /home/${USER_NAME} &&\
    chown ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME} &&\
    chown -R ${USER_NAME}:${GROUP_NAME} /cas/

USER ${USER_NAME}

ENTRYPOINT ["python", "-m", "clip_server"]