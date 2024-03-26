#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-26 17:50
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


ROOT_DIRPATH=/home/zxyang/YBD
HF_CACHE_DIRPATH=/zxyang/HF/Cache
CACHE_DIRPATH=${ROOT_DIRPATH}/Temp
MODEL_ID_FILEPATH=${ROOT_DIRPATH}/model.ids
PROCESS_FLAG_FILEPATH=${ROOT_DIRPATH}/process.flg
DATASET_DIRPATH=${ROOT_DIRPATH}/dataset

python -m youngbench.dataset.construct.builder \
  --version 0.0.1 \
  --hf-cache-dirpath ${HF_CACHE_DIRPATH} \
  --convert-cache-dirpath ${CACHE_DIRPATH} \
  --model-id-filepath ${MODEL_ID_FILEPATH} \
  --save-dirpath ${DATASET_DIRPATH} \
  --process-flag-path ${PROCESS_FLAG_FILEPATH} \
  --logging-path ${ROOT_DIRPATH}/convert.log \
  --device 'cpu' \
  --mode 'Create'
