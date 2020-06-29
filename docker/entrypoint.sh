#!/bin/sh
bert-serving-start -http_port 80 -num_worker=4 -model_dir /app/model -verbose
