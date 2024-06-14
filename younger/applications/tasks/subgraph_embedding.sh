#!/bin/bash

python subgraph_embedding.py --dataset-dir /younger/younger/silly_splited_filtered_instances/np \
  --save-dir /younger/final_experiments/subgraph_embedding \
  --baseline-model gcn \
  --encode-type operator \
  --checkpoint-filepath /younger/final_experiments/final_log/node_prediction/selected_checkpoint/gcn_np_operator_Epoch_194_Step_2320.cp

python subgraph_embedding.py --dataset-dir /younger/younger/silly_splited_filtered_instances/np \
  --save-dir /younger/final_experiments/subgraph_embedding \
  --baseline-model gcn \
  --encode-type node \
  --checkpoint-filepath /younger/final_experiments/final_log/node_prediction/selected_checkpoint/gcn_np_node_Epoch_195_Step_3120.cp

python subgraph_embedding.py --dataset-dir /younger/younger/silly_splited_filtered_instances/np \
  --save-dir /younger/final_experiments/subgraph_embedding \
  --baseline-model gat \
  --encode-type operator \
  --checkpoint-filepath /younger/final_experiments/final_log/node_prediction/selected_checkpoint/gat_np_operator_Epoch_189_Step_2260.cp

python subgraph_embedding.py --dataset-dir /younger/younger/silly_splited_filtered_instances/np \
  --save-dir /younger/final_experiments/subgraph_embedding \
  --baseline-model sage \
  --encode-type operator \
  --checkpoint-filepath /younger/final_experiments/final_log/node_prediction/selected_checkpoint/sage_np_operator_Epoch_117_Step_1400.cp

python subgraph_embedding.py --dataset-dir /younger/younger/silly_splited_filtered_instances/np \
  --save-dir /younger/final_experiments/subgraph_embedding \
  --baseline-model sage \
  --encode-type node \
  --checkpoint-filepath /younger/final_experiments/final_log/node_prediction/selected_checkpoint/sage_np_node_Epoch_167_Step_2660.cp
