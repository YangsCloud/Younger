#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

HF_TOKEN=SECRET

TASK=Tasks-No-0

ROOT_DIRPATH=/home/jason/Files/YBD
CONVERT_CACHE_DIRPATH=${ROOT_DIRPATH}/ConvertCache
PROCESS_FLAG_FILEPATH=${ROOT_DIRPATH}/Process-${TASK}.flg
INSTANCES_DIRPATH=${ROOT_DIRPATH}/instances
LOGGING_FILEPATH=${ROOT_DIRPATH}/Convert-${TASK}.log

HF_CACHE_DIRPATH=/home/jason/Files/HF/Cache

MODEL_ID_FILEPATH=/home/jason/Files/Neats/${TASK}.json

python -m youngbench.dataset.construct.builder \
  --version 0.0.1 \
  --hf-cache-dirpath ${HF_CACHE_DIRPATH} \
  --convert-cache-dirpath ${CONVERT_CACHE_DIRPATH} \
  --model-id-filepath ${MODEL_ID_FILEPATH} \
  --save-dirpath ${INSTANCES_DIRPATH} \
  --process-flag-path ${PROCESS_FLAG_FILEPATH} \
  --logging-path ${LOGGING_FILEPATH} \
  --hf-token ${HF_TOKEN} \
  --device 'cpu'
