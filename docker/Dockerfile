FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN pip install bert-serving-server[http]
COPY ./docker/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
WORKDIR /app
RUN curl -o model.zip https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip && \
unzip model.zip && mv uncased_L-12_H-768_A-12 model && rm model.zip &&ls -al
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
HEALTHCHECK --timeout=5s CMD curl -f http://localhost:80/status/server || exit 1
