FROM tensorflow/tensorflow:1.12.0-gpu-py3
COPY ./ /app
COPY ./docker/entrypoint.sh /app
WORKDIR /app
RUN pip install -r requirements.gpu.txt
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []

