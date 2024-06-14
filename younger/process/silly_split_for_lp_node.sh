#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

NODE_LB=1
EDGE_LB=1

FILTERED_INSTANCE='/younger/younger/filtered_instances/2024-06-03-13-55-13'
DATETIME_STR=$(date +%Y-%m-%d-%H-%M-%S)

younger datasets split \
  --tasks-filepath ./huggingface_tasks.json --dataset-dirpath ${FILTERED_INSTANCE} --save-dirpath /younger/final_experiments/silly_splited_filtered_instances_for_lp_np_node/${DATETIME_STR}/ \
  --version initial_nl${NODE_LB}_el${EDGE_LB} \
  --silly \
  --node-size-lbound $NODE_LB \
  --edge-size-lbound $EDGE_LB \
  --train-proportion 80 --valid-proportion 10 --test-proportion 10 \
  --partition-number 10 \
  --worker-number 1 \
  --seed 1234 \
  --logging-filepath ./silly_splited_filtered_instances_for_lp_np_node${NODE_LB}_el${EDGE_LB}_${DATETIME_STR}.log

