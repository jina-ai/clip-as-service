#!/bin/sh
num_worker=$1
bert-serving-start -num_worker=${num_worker} -model_dir /model
