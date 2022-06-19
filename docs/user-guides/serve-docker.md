# Serve in Docker Container

You can serve `clip_serve` inside a Docker container. We provide a Dockerfile in the repository, which is CUDA-enabled with optimized package installation. 

## Build

```bash
git clone https://github.com/jina-ai/clip-as-service.git
docker build . -f Dockerfiles/server.Dockerfile  --build-arg GROUP_ID=$(id -g ${USER}) --build-arg USER_ID=$(id -u ${USER}) -t jinaai/clip-as-service
```

```{tip}
The build argument `--build-arg GROUP_ID=$(id -g ${USER}) --build-arg USER_ID=$(id -u ${USER})` is optional, but having them is highly recommended as it allows you to reuse host's cache with the correct access.
```


## Run

```bash
docker run -p 51009:51000 -v $HOME/.cache:/home/cas/.cache --gpus all jinaai/clip-as-service
```

Here, `51009` is the public port on the host and `51000` is the {ref}`in-container port defined inside YAML<flow-config>`.

Due to the limitation of the terminal inside Docker container, you will **not** see the classic Jina progress bar on start. Instead, you will face a few minutes awkward silent while model downloading and then see "Flow is ready to serve" dialog.

The CLI usage is the same as described here {ref}`start-server`.

```{tip}
The argument `-v $HOME/.cache:/home/cas/.cache` leverages host's cache and prevents you to download the same model next time on start. 
```