#!/bin/bash

./train_gcn_lp_node.sh

./train_gcn_lp_operator.sh

./train_gat_lp_node.sh 

./train_gat_lp_operator.sh

./train_sage_lp_node.sh

./train_sage_lp_operator.sh

echo "All scripts have completed."

