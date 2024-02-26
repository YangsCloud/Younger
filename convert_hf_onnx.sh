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

python -m youngbench.dataset.scripts.convert_hf_onnx --key likes --info-dirpath ${DFS_PATH} --save-dirpath ${DFS_PATH} --save-flag-path ${HF_SAVE_FLAG_PATH} --cache-dirpath ${HF_CACHE_PATH} --cache-flag-path ${HF_CACHE_FLAG_PATH}
