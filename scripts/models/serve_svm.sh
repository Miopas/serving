#!/bin/bash

set -e  -x
# 设置模型路径
TESTDATA="/tmp/serving/"

MODEL_BASE_PATH="/models"

# 设置模型名称
MODEL_NAME=svm_cls

PORT=8501

# 启动服务
docker run -t --rm -p 8501:8501 -p 8500:8500 -v $(pwd):$(pwd) -v ${TESTDATA}:${MODEL_BASE_PATH} tensorflow/serving:nightly-devel-mypy2 \
	bash -c "tensorflow_model_server \
		--port=8500 --rest_api_port=8501 \
		--model_name=${MODEL_NAME} \
		--model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}" &

#docker run -t --rm -p 8501:8501 -p 8500:8500 -v ${TESTDATA}:${MODEL_BASE_PATH} tensorflow/serving:nightly-devel-mypy2 \
#	bash -c "tensorflow_model_server \
#			--port=8500 --rest_api_port=8501 \
#			--model_config_file=/models/models.config \
#			--model_config_file_poll_wait_seconds=60 \
#		" &


#docker run -t --rm --name tf_serving_svm_cls -p 8501:8501 -p 8500:8500 -v $(pwd):$(pwd) -v ${TESTDATA}:${MODEL_BASE_PATH} tensorflow/serving:nightly-devel-mypy2 \
#	bash -c "tensorflow_model_server \
#			--port=8500 --rest_api_port=8501 \
#			--model_config_file=/models/models.config \
#		" &
