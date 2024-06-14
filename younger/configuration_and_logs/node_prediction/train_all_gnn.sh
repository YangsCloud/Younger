#!/bin/bash

./train_gcn_np_operator.sh

./train_gat_np_operator.sh

./train_sage_np_operator.sh

./train_gcn_np_node.sh

# ./train_gat_np_node.sh

./train_sage_np_node.sh

echo "All scripts have completed."
