#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


NODE_LB=30
EDGE_LB=30

younger datasets split \
  --tasks-filepath ./huggingface_tasks.json --dataset-dirpath ./filtered_dataset/ --save-dirpath ./splited_filtered_dataset/ \
  --version initial_nl${NODE_LB}_el${EDGE_LB}_v0.0.1 \
  --node-size-lbound $NODE_LB \
  --edge-size-lbound $EDGE_LB \
  --train-proportion 80 --valid-proportion 10 --test-proportion 10 \
  --partition-number 10 \
  --worker-number 32 \
  --seed 1234 \
  --logging-filepath ./split_filtered_dataset_nl${NODE_LB}_el${EDGE_LB}.log
