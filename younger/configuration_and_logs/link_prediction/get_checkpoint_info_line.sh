#!/bin/bash

python get_checkpoint_info_line.py \
  --log-file-path /root/autodl-tmp/Experiments/Link_prediction/gat_lp_node.log \
  --save-dir /root/autodl-tmp/Experiments/Link_prediction/infos \

python get_checkpoint_info_line.py \
  --log-file-path /root/autodl-tmp/Experiments/Link_prediction/gat_lp_operator.log \
  --save-dir /root/autodl-tmp/Experiments/Link_prediction/infos \

python get_checkpoint_info_line.py \
  --log-file-path /root/autodl-tmp/Experiments/Link_prediction/gcn_lp_node.log \
  --save-dir /root/autodl-tmp/Experiments/Link_prediction/infos \

python get_checkpoint_info_line.py \
  --log-file-path /root/autodl-tmp/Experiments/Link_prediction/gcn_lp_operator.log \
  --save-dir /root/autodl-tmp/Experiments/Link_prediction/infos \

python get_checkpoint_info_line.py \
  --log-file-path /root/autodl-tmp/Experiments/Link_prediction/sage_lp_node.log \
  --save-dir /root/autodl-tmp/Experiments/Link_prediction/infos \

python get_checkpoint_info_line.py \
  --log-file-path /root/autodl-tmp/Experiments/Link_prediction/sage_lp_operator.log \
  --save-dir /root/autodl-tmp/Experiments/Link_prediction/infos \