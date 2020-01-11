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
docker run -t --rm -p ${PORT}:${PORT} -v $(pwd):$(pwd) -v ${TESTDATA}:${MODEL_BASE_PATH} tensorflow/serving:nightly-devel-tf1.5 \
	bash -c "$(pwd)/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
		--port=8500 --rest_api_port=8501 \
		--model_name=${MODEL_NAME} \
		--model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}" &
