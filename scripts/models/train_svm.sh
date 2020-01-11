#!/bin/bash
set -e  -x
docker run -t -v $(pwd):$(pwd) -v /tmp:/tmp tensorflow/serving:nightly-devel-mypy2 \
	bash -c "cd $(pwd)/tf-svm; python train.py \
                --train_data_file=./data/sample.csv  \
                --dev_data_file=./data/sample.csv \
                --export_path_base=/tmp/serving/svm_cls \
                --num_class=5  \
                --batch_size=10 \
                --num_epochs=1 \
                --evaluate_every=1 \
                --checkpoint_every=1 \
                --num_checkpoints=1 \
                --regulation_rate=1e-4 \
                --model_version=1" &

