#!/bin/bash

THIS_NAME=gat_np_operator

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_DIRPATH=./checkpoint-gat-np/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
# CHECKPOINT_FILEPATH=${THIS_NAME}/
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning train \
  --task-name node_prediciton \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-dirpath ${CHECKPOINT_DIRPATH} --checkpoint-name ${CHECKPOINT_NAME} --keep-number 200 \
  --train-batch-size 512 --valid-batch-size 512 --shuffle \
  --life-cycle 200 --report-period 10 --update-period 1 --train-period 20 --valid-period 20 \
  --device GPU \
  --world-size 4 --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} --master-rank ${MASTER_RANK} \
  --seed 12345
#  --make-deterministic 
#  --checkpoint-filepath ${CHECKPOINT_FILEPATH} --reset-optimizer --reset-period \

