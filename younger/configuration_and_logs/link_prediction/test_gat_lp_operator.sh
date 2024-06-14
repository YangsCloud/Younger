#!/bin/bash

THIS_NAME=gat_lp_operator

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_DIRPATH=./checkpoint-gat-lp/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
CHECKPOINT_FILEPATH=/root/autodl-tmp/Experiments/Link_prediction/checkpoint-gat-lp/gat_lp_operator/gat_lp_operator_Epoch_3_Step_4200.cp
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning test \
  --task-name link_prediction \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-filepath ${CHECKPOINT_FILEPATH} \
  --test-batch-size 1 \
  --device GPU \

