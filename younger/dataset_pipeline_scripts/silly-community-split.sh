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
    echo "    ./silly-community-split.sh {filtered_instances_dirpath}"
    exit 1
fi

NODE_LB=1
EDGE_LB=1

FILTERED_INSTANCE=${1}
DATETIME_STR=$(date +%Y-%m-%d-%H-%M-%S)

younger datasets split \
  --tasks-filepath ./huggingface_tasks.json --dataset-dirpath ${FILTERED_INSTANCE} --save-dirpath ../silly_splited_filtered_instances/${DATETIME_STR}/ \
  --version initial_nl${NODE_LB}_el${EDGE_LB}_nu${NODE_UB}_eu${EDGE_UB} \
  --silly \
  --community \
  --node-size-lbound $NODE_LB \
  --edge-size-lbound $EDGE_LB \
  --train-proportion 80 --valid-proportion 10 --test-proportion 10 \
  --partition-number 10 \
  --worker-number 32 \
  --seed 16861 \
  --logging-filepath ./silly_split_filtered_dataset_nl${NODE_LB}_el${EDGE_LB}_${DATETIME_STR}.log

