#!/bin/bash

python draw_subgraph_embedding.py --subgraph-embedding-dir /younger/final_experiments/subgraph_embedding \
  --save-dir /younger/final_experiments/visualization/subgraph_embedding \
  --embedding-model gcn \
  --encode-type operator

python draw_subgraph_embedding.py --subgraph-embedding-dir /younger/final_experiments/subgraph_embedding \
  --save-dir /younger/final_experiments/visualization/subgraph_embedding \
  --embedding-model gcn \
  --encode-type node

python draw_subgraph_embedding.py --subgraph-embedding-dir /younger/final_experiments/subgraph_embedding \
  --save-dir /younger/final_experiments/visualization/subgraph_embedding \
  --embedding-model gat \
  --encode-type operator

python draw_subgraph_embedding.py --subgraph-embedding-dir /younger/final_experiments/subgraph_embedding \
  --save-dir /younger/final_experiments/visualization/subgraph_embedding \
  --embedding-model sage \
  --encode-type operator

python draw_subgraph_embedding.py --subgraph-embedding-dir /younger/final_experiments/subgraph_embedding \
  --save-dir /younger/final_experiments/visualization/subgraph_embedding \
  --embedding-model sage \
  --encode-type node