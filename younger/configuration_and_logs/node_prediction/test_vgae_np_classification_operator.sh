#!/bin/bash

THIS_NAME=vgae_np_classification_operator

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_FILEPATH=/root/autodl-tmp/Experiments/Node_prediction/selected_checkpoint/vgae_np_classification_operator_Epoch_75_Step_3500.cp
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning test \
  --task-name node_prediciton \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-filepath ${CHECKPOINT_FILEPATH} \
  --test-batch-size 512 \
  --device GPU \
