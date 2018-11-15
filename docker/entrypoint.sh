#!/bin/sh
num_worker=$1
python app.py -num_worker=${num_worker} -model_dir /model 
