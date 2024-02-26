#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-02-24 17:19
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

. ./constants.sh

python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby downloads --filter ${DFS_TAG} --part-size 10000 --save-dirpath ${DFS_PATH} --logging-path ${HF_LOG_PATH}/${DFS_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby downloads --filter ${TFS_TAG} --part-size 10000 --save-dirpath ${TFS_PATH} --logging-path ${HF_LOG_PATH}/${TFS_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby downloads --filter ${STFS_TAG} --part-size 10000 --save-dirpath ${STFS_PATH} --logging-path ${HF_LOG_PATH}/${STFS_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby downloads --filter ${TIMM_TAG} --part-size 10000 --save-dirpath ${TIMM_PATH} --logging-path ${HF_LOG_PATH}/${TIMM_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby downloads --filter ${ONNX_TAG} --part-size 10000 --save-dirpath ${ONNX_PATH} --logging-path ${HF_LOG_PATH}/${ONNX_TAG}.log

python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby likes --filter ${DFS_TAG} --part-size 10000 --save-dirpath ${DFS_PATH} --logging-path ${HF_LOG_PATH}/${DFS_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby likes --filter ${TFS_TAG} --part-size 10000 --save-dirpath ${TFS_PATH} --logging-path ${HF_LOG_PATH}/${TFS_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby likes --filter ${STFS_TAG} --part-size 10000 --save-dirpath ${STFS_PATH} --logging-path ${HF_LOG_PATH}/${STFS_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby likes --filter ${TIMM_TAG} --part-size 10000 --save-dirpath ${TIMM_PATH} --logging-path ${HF_LOG_PATH}/${TIMM_TAG}.log
python -m youngbench.dataset.scripts.get_hf_models_info --full --config --sort --sortby likes --filter ${ONNX_TAG} --part-size 10000 --save-dirpath ${ONNX_PATH} --logging-path ${HF_LOG_PATH}/${ONNX_TAG}.log
