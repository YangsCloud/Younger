#!/bin/bash

python get_checkpoint_info_line.py \
  --log-file-path ./gat_np_operator.log \
  --save-dir ./infos \

python get_checkpoint_info_line.py \
  --log-file-path ./gcn_np_node.log \
  --save-dir ./infos \

python get_checkpoint_info_line.py \
  --log-file-path ./gcn_np_operator.log \
  --save-dir ./infos \

python get_checkpoint_info_line.py \
  --log-file-path ./sage_np_node.log \
  --save-dir ./infos \

python get_checkpoint_info_line.py \
  --log-file-path ./sage_np_operator.log \
  --save-dir ./infos \

python get_checkpoint_info_line.py \
  --log-file-path ./gae_np_classification_operator.log \
  --save-dir ./infos \
  
python get_checkpoint_info_line.py \
  --log-file-path ./gae_np_classification_node.log \
  --save-dir ./infos \

python get_checkpoint_info_line.py \
  --log-file-path ./vgae_np_classification_operator.log \
  --save-dir ./infos \

python get_checkpoint_info_line.py \
  --log-file-path ./vgae_np_classification_node.log \
  --save-dir ./infos \