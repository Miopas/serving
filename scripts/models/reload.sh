#!/bin/bash
TESTDATA="/tmp/serving/"
MODEL_BASE_PATH="/models"
docker exec -t tf_serving_svm_cls \
	bash -c "cd $(pwd)/tf-svm; python grpc_reload.py http://localhost:8500" &
