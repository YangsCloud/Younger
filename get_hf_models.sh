#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-02-24 23:28
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

. ./constants.sh

#python -m youngbench.dataset.construct.get_hf_models --disk-usage-percentage 0.9 --model-ids-filepath ${HF_PATH}/70K-1-Neat.json --cache-dirpath ${HF_CACHE_PATH} --cache-tar-dirpath ${HF_CACHE_TAR_PATH} --cache-flag-path ${HF_CACHE_FLAG_PATH} --fails-flag-path ${HF_FAILS_FLAG_PATH} --hf-token ${HF_TOKEN} --logging-path ${HF_LOG_PATH}/70K-1-Neat.log --yes
python -m youngbench.dataset.construct.get_hf_models --disk-usage-percentage 0.9 --model-ids-filepath ${HF_PATH}/70K-2-Neat.json --cache-dirpath ${HF_CACHE_PATH} --cache-tar-dirpath ${HF_CACHE_TAR_PATH} --cache-flag-path ${HF_CACHE_FLAG_PATH} --fails-flag-path ${HF_FAILS_FLAG_PATH} --hf-token ${HF_TOKEN} --logging-path ${HF_LOG_PATH}/70K-2-Neat.log --yes
#python -m youngbench.dataset.construct.get_hf_models --disk-usage-percentage 0.9 --model-ids-filepath ${HF_PATH}/70K-3-Neat.json --cache-dirpath ${HF_CACHE_PATH} --cache-tar-dirpath ${HF_CACHE_TAR_PATH} --cache-flag-path ${HF_CACHE_FLAG_PATH} --fails-flag-path ${HF_FAILS_FLAG_PATH} --hf-token ${HF_TOKEN} --logging-path ${HF_LOG_PATH}/70K-3-Neat.log --yes
#python -m youngbench.dataset.construct.get_hf_models --disk-usage-percentage 0.9 --model-ids-filepath ${HF_PATH}/70K-4-Neat.json --cache-dirpath ${HF_CACHE_PATH} --cache-tar-dirpath ${HF_CACHE_TAR_PATH} --cache-flag-path ${HF_CACHE_FLAG_PATH} --fails-flag-path ${HF_FAILS_FLAG_PATH} --hf-token ${HF_TOKEN} --logging-path ${HF_LOG_PATH}/70K-4-Neat.log --yes

# python -m youngbench.dataset.construct.get_hf_models --model-ids-filepath ~/WSL_Share/70K-3-Neat.json --cache-dirpath ${HF_CACHE_PATH} --cache-flag-path ${HF_CACHE_FLAG_PATH}  --fails-flag-path ${HF_FAILS_FLAG_PATH}
