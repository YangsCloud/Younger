#!/bin/bash

THIS_NAME=sage_np_node

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_FILEPATH=/root/autodl-tmp/Experiments/Node_prediction/selected_checkpoint/sage_np_node_Epoch_167_Step_2660.cp
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning test \
  --task-name node_prediciton \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-filepath ${CHECKPOINT_FILEPATH} \
  --test-batch-size 512 \
  --device GPU \
