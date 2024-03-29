#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-02-24 23:28
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if [ "$#" -ne 1 ]; then
    echo "Error: Need Exact 1 Argument"
    exit 1
fi

HF_TOKEN=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

TASK=Tasks-No-${1}

ROOT_DIRPATH=/home/jason/Files/YBD
PROCESS_FLAG_FILEPATH=${ROOT_DIRPATH}/Process-${TASK}.flg
HF_CACHE_FLAG_PATH=${ROOT_DIRPATH}/cache.flg
HF_FAILS_FLAG_PATH=${ROOT_DIRPATH}/fails.flg
LOGGING_FILEPATH=${ROOT_DIRPATH}/Download-${TASK}.log

HF_CACHE_DIRPATH=/home/jason/Files/HF/Cache

python -m youngbench.dataset.construct.get_large_hf_models --disk-usage-percentage 0.9 --process-flag-path ${PROCESS_FLAG_FILEPATH} --cache-dirpath ${HF_CACHE_DIRPATH} --cache-flag-path ${HF_CACHE_FLAG_PATH} --fails-flag-path ${HF_FAILS_FLAG_PATH} --hf-token ${HF_TOKEN} --logging-path ${LOGGING_FILEPATH} --yes
