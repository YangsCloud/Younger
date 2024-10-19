#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if [ "$#" -ne 1 ]; then
    echo "Usage:"
    echo "    ./silly-split-full.sh {filtered_instances_dirpath}"
    exit 1
fi


NODE_LB=1
EDGE_LB=1

FILTERED_INSTANCE=${1}
DATETIME_STR=$(date +%Y-%m-%d-%H-%M-%S)

younger datasets split \
  --tasks-filepath ./huggingface_tasks.json --dataset-dirpath ${FILTERED_INSTANCE} --save-dirpath ../silly_splited_filtered_instances/${DATETIME_STR}/ \
  --version initial_full \
  --silly \
  --node-size-lbound $NODE_LB \
  --edge-size-lbound $EDGE_LB \
  --train-proportion 100 --valid-proportion 0 --test-proportion 0 \
  --partition-number 10 \
  --worker-number 28 \
  --seed 1234 \
  --logging-filepath ./silly_split_filtered_dataset_full.log
