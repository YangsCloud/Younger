#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FILTER_DIR=../YoungBench

DATETIME_STR=$(date +%Y-%m-%d-%H-%M-%S)

younger datasets filter \
  --dataset-dirpath ../all_instances/ \
  --save-dirpath ${FILTER_DIR}/${DATETIME_STR} \
  --clean \
  --worker-number 20 \
  --logging-filepath ${FILTER_DIR}/filter_all_instances_${DATETIME_STR}.log
