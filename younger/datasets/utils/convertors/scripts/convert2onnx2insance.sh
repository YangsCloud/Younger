#!/bin/bash

JSON_PATH="/Users/zrsion/Younger/model_id_list.json"
OUTPUT_PARENT_DIR="/Users/zrsion/Younger/Test_onnx2instance"

tasks=($(jq -r 'keys[]' "$JSON_PATH"))
model_ids=($(jq -r '.[]' "$JSON_PATH"))

for ((i=0; i<${#tasks[@]}; i++)); do
    task=${tasks[i]}
    model_id=${model_ids[i]}
    python younger/datasets/utils/convertors/scripts/convert.py --quantize --task "$task" --model_id "$model_id" --trust_remote_code --skip_validation --output_parent_dir "$OUTPUT_PARENT_DIR" --remove_other_files
done