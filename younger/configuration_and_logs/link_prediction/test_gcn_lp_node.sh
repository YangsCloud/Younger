#!/bin/bash

THIS_NAME=gcn_lp_node

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_DIRPATH=./checkpoint-gcn-lp/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
CHECKPOINT_FILEPATH=/root/autodl-tmp/Experiments/Link_prediction/checkpoint-gcn-lp/gcn_lp_node/gcn_lp_node_Epoch_1_Step_1000.cp
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning test \
  --task-name link_prediction \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-filepath ${CHECKPOINT_FILEPATH} \
  --test-batch-size 1 \
  --device GPU \

