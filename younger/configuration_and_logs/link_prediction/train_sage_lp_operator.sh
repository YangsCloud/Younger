#!/bin/bash

THIS_NAME=sage_lp_operator

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_DIRPATH=./checkpoint-sage-lp/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
# CHECKPOINT_FILEPATH=${THIS_NAME}/
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning train \
  --task-name link_prediction \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-dirpath ${CHECKPOINT_DIRPATH} --checkpoint-name ${CHECKPOINT_NAME} --keep-number 200 \
  --train-batch-size 1 --valid-batch-size 1 --shuffle \
  --life-cycle 6 --report-period 50 --update-period 1 --train-period 200 --valid-period 200 \
  --device GPU \
  --world-size 4 --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} --master-rank ${MASTER_RANK} \
  --seed 12345
#  --make-deterministic 
#  --checkpoint-filepath ${CHECKPOINT_FILEPATH} --reset-optimizer --reset-period \

