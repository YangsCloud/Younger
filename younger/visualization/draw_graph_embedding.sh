#!/bin/bash

python draw_graph_embedding.py --graph-embedding-dir /younger/final_experiments/graph_embedding \
  --save-dir /younger/final_experiments/visualization/graph_embedding \
  --embedding-model gcn \
  --encode-type operator 

python draw_graph_embedding.py --graph-embedding-dir /younger/final_experiments/graph_embedding \
  --save-dir /younger/final_experiments/visualization/graph_embedding \
  --embedding-model gcn \
  --encode-type node 

python draw_graph_embedding.py --graph-embedding-dir /younger/final_experiments/graph_embedding \
  --save-dir /younger/final_experiments/visualization/graph_embedding \
  --embedding-model gat \
  --encode-type operator 

python draw_graph_embedding.py --graph-embedding-dir /younger/final_experiments/graph_embedding \
  --save-dir /younger/final_experiments/visualization/graph_embedding \
  --embedding-model sage \
  --encode-type operator 

python draw_graph_embedding.py --graph-embedding-dir /younger/final_experiments/graph_embedding \
  --save-dir /younger/final_experiments/visualization/graph_embedding \
  --embedding-model sage \
  --encode-type node 