#!/bin/bash

set -e  -x

MODEL_NAME=svm_cls
cd $(pwd)/tf-svm;python client.py http://localhost:8501/v1/models/${MODEL_NAME}:predict 
