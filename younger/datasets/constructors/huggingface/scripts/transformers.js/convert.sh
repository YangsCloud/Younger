#!/bin/bash
#
# Copyright (c) Luzhou Peng (彭路洲) & Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-09 14:03
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


if [ "$#" -ne 3 ]; then
    echo "Usage:"
    echo "    ./convert.sh {model_ids_filepath} {cache_dirpath} {save_dirpath}"
    exit 1
fi

MODEL_IDS_FILEPATH=${1}
CACHE_DIRPATH=${2}
SAVE_DIRPATH=${3}

model_ids=($(jq -r '.[]' "${MODEL_IDS_FILEPATH}"))

for ((i=0; i<${#model_ids[@]}; i++)); do
    model_id=${model_ids[i]}
    python -m convert.main --quantize --model_id "${model_id}" --trust_remote_code --skip_validation --onnx_output_dir "${CACHE_DIRPATH}" --instance_output_dir "${SAVE_DIRPATH}" --clean_onnx_dir
done
