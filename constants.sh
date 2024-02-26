#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-02-24 23:43
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DFS_TAG=diffusers
TFS_TAG=transformers
STFS_TAG=sentence-transformers
TIMM_TAG=timm
ONNX_TAG=onnx

HF_PATH=/mnt/d/HF
DFS_PATH=${HF_PATH}/Diffusers
TFS_PATH=${HF_PATH}/Transformers
STFS_PATH=${HF_PATH}/Sentence_Transformers
TIMM_PATH=${HF_PATH}/Timm
ONNX_PATH=${HF_PATH}/ONNX

mkdir -p ${DFS_PATH}
mkdir -p ${TFS_PATH}
mkdir -p ${STFS_PATH}
mkdir -p ${TIMM_PATH}
mkdir -p ${ONNX_PATH}

HF_LOG_PATH=${HF_PATH}/logs
mkdir -p ${HF_LOG_PATH}

HF_SAVE_FLAG_PATH=${HF_PATH}/save.flg
HF_CACHE_FLAG_PATH=${HF_PATH}/cache.flg
HF_CACHE_PATH=${HF_PATH}/Cache
