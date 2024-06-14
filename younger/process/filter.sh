#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FILTER_DIR=/younger/younger/filter_all_instances_with_model_id

DATETIME_STR=$(date +%Y-%m-%d-%H-%M-%S)

younger datasets filter \
  --dataset-dirpath /younger/younger/all_instances \
  --save-dirpath ${FILTER_DIR}/${DATETIME_STR} \
  --worker-number 30 \
  --logging-filepath ${FILTER_DIR}/${DATETIME_STR}.log
