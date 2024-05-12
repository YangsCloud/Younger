#!/bin/bash
#
# Copyright (c) Luzhou Peng (彭路洲) & Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-09 14:03
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


if [ "$#" -ne 2 ]; then
    echo "Error: Need Exact 2 Argument"
    exit 1
fi

MODEL_IDS_FILEPATH=${1}
SAVE_DIRPATH=${2}

tasks=($(jq -r 'keys[]' "${MODEL_IDS_FILEPATH}"))
model_ids=($(jq -r '.[]' "${MODEL_IDS_FILEPATH}"))

for ((i=0; i<${#tasks[@]}; i++)); do
    task=${tasks[i]}
    model_id=${model_ids[i]}
    python younger/datasets/utils/convertors/scripts/convert.py --quantize --task "$task" --model_id "$model_id" --trust_remote_code --skip_validation --output_parent_dir "${SAVE_DIRPATH}" --remove_other_files
done