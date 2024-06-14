
#!/bin/bash

THIS_NAME=gae_np_classification_operator

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_DIRPATH=./checkpoint-gae-np/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
# CHECKPOINT_FILEPATH=${THIS_NAME}/
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUDA_LAUNCH_BLOCKING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 younger applications deep_learning train \
	--task-name node_prediciton \
	--config-filepath ${CONFIG_FILEPATH} \
	--checkpoint-dirpath ${CHECKPOINT_DIRPATH} --checkpoint-name ${CHECKPOINT_NAME} --keep-number 100 \
	--train-batch-size 512 --valid-batch-size 512 --shuffle \
	--life-cycle 100 --report-period 10 --update-period 1 --train-period 100 --valid-period 100 \
	--device GPU \
	--world-size 1 --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} --master-rank ${MASTER_RANK} \
	--seed 12345
#  --make-deterministic 
#  --checkpoint-filepath ${CHECKPOINT_FILEPATH} --reset-optimizer --reset-period \
