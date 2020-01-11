#!/bin/bash

PORT=8500

set -e  -x

# 设置模型路径
TESTDATA="/tmp"

MODEL_BASE_PATH="/models"

# 设置模型名称
MODEL_NAME=mnist

# 启动服务
# 挂载了当前目录，使用 bazel-bin 目录下的新的 bin 来启动
docker run -t -v $(pwd):$(pwd) -v ${TESTDATA}:${MODEL_BASE_PATH} tensorflow/serving \
	bash -c "python $(pwd)/tensorflow_serving/example/mnist_client.py --num_tests=1 --server=127.0.0.0:8500" &
