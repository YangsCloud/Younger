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

# FILTERED_INSTANCE=/younger/younger/filter_all_instances_with_model_id/operator/2024-06-08-15-28-08
# DATETIME_STR=$(date +%Y-%m-%d-%H-%M-%S)

# younger datasets split \
#   --tasks-filepath ./huggingface_tasks.json --dataset-dirpath ${FILTERED_INSTANCE} --save-dirpath /younger/younger/dataset_graph_embedding/operator/${DATETIME_STR}/ \
#   --version initial_full \
#   --silly \
#   --node-size-lbound $NODE_LB \
#   --edge-size-lbound $EDGE_LB \
#   --train-proportion 100 --valid-proportion 0 --test-proportion 0 \
#   --partition-number 10 \
#   --worker-number 28 \
#   --seed 1234 \
#   --logging-filepath ./silly_split_filtered_dataset_graph_embedding_full_new.log

FILTERED_INSTANCE=/younger/younger/filter_all_instances_with_model_id/node/2024-06-08-11-43-23
DATETIME_STR=$(date +%Y-%m-%d-%H-%M-%S)

younger datasets split \
  --tasks-filepath ./huggingface_tasks.json --dataset-dirpath ${FILTERED_INSTANCE} --save-dirpath /younger/younger/dataset_graph_embedding/node/${DATETIME_STR}/ \
  --version initial_full \
  --silly \
  --node-size-lbound $NODE_LB \
  --edge-size-lbound $EDGE_LB \
  --train-proportion 100 --valid-proportion 0 --test-proportion 0 \
  --partition-number 10 \
  --worker-number 24 \
  --seed 1234 \
  --logging-filepath ./silly_split_filtered_dataset_graph_embedding_full_new.log