#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET_DIRPATH=./filtered_dataset
SAVE_DIRPATH=./statistics_filtered_dataset

younger datasets statistics \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath ${SAVE_DIRPATH} \
    --tasks \
    --dataset-names \
    --dataset-splits \
    --metric-names \
    --worker-number 32 \
    --logging-filepath ./statistics_filtered_dataset.log

younger datasets statistics \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath ${SAVE_DIRPATH} \
    --tasks \
    --dataset-names \
    --dataset-splits 'test' \
    --metric-names 'accuracy' \
    --worker-number 32 \
    --logging-filepath ./statistics_filtered_dataset.log

younger datasets statistics \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath ${SAVE_DIRPATH} \
    --tasks \
    --dataset-names \
    --dataset-splits 'test' \
    --metric-names 'f1' \
    --worker-number 32 \
    --logging-filepath ./statistics_filtered_dataset.log

younger datasets statistics \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath ${SAVE_DIRPATH} \
    --tasks \
    --dataset-names \
    --dataset-splits 'test' \
    --metric-names 'recall' \
    --worker-number 32 \
    --logging-filepath ./statistics_filtered_dataset.log

younger datasets statistics \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath ${SAVE_DIRPATH} \
    --tasks \
    --dataset-names \
    --dataset-splits 'test' \
    --metric-names 'precision' \
    --worker-number 32 \
    --logging-filepath ./statistics_filtered_dataset.log

younger datasets statistics \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath ${SAVE_DIRPATH} \
    --tasks \
    --dataset-names \
    --dataset-splits 'test' \
    --metric-names 'wer' \
    --worker-number 32 \
    --logging-filepath ./statistics_filtered_dataset.log
