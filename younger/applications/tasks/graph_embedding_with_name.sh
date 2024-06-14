#!/bin/bash

python graph_embedding.py --subgraph-dir /younger/final_experiments/subgraph_embedding \
  --graph-dir /younger/younger/dataset_graph_embedding/operator/2024-06-12-10-14-59/initial_full/train/graph \
  --save-dir /younger/final_experiments/graph_embedding \
  --block-get-type greedy_modularity_communities \
  --embedding-model gcn \
  --encode-type operator \
  --worker-number 32

python graph_embedding.py --subgraph-dir /younger/final_experiments/subgraph_embedding \
  --graph-dir /younger/younger/dataset_graph_embedding/node/2024-06-12-13-39-24/initial_full/train/graph \
  --save-dir /younger/final_experiments/graph_embedding \
  --block-get-type greedy_modularity_communities \
  --embedding-model gcn \
  --encode-type node \
  --worker-number 32

python graph_embedding.py --subgraph-dir /younger/final_experiments/subgraph_embedding \
  --graph-dir /younger/younger/dataset_graph_embedding/operator/2024-06-12-10-14-59/initial_full/train/graph \
  --save-dir /younger/final_experiments/graph_embedding \
  --block-get-type greedy_modularity_communities \
  --embedding-model gat \
  --encode-type operator \
  --worker-number 32

python graph_embedding.py --subgraph-dir /younger/final_experiments/subgraph_embedding \
  --graph-dir /younger/younger/dataset_graph_embedding/operator/2024-06-12-10-14-59/initial_full/train/graph \
  --save-dir /younger/final_experiments/graph_embedding \
  --block-get-type greedy_modularity_communities \
  --embedding-model sage \
  --encode-type operator \
  --worker-number 32

python graph_embedding.py --subgraph-dir /younger/final_experiments/subgraph_embedding \
  --graph-dir /younger/younger/dataset_graph_embedding/node/2024-06-12-13-39-24/initial_full/train/graph \
  --save-dir /younger/final_experiments/graph_embedding \
  --block-get-type greedy_modularity_communities \
  --embedding-model sage \
  --encode-type node \
  --worker-number 32