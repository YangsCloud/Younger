#!/bin/bash

./test_gcn_np_operator.sh

./test_gat_np_operator.sh

./test_sage_np_operator.sh

./test_gcn_np_node.sh

./test_sage_np_node.sh

./test_gae_np_classification_node.sh

./test_gae_np_classification_operator.sh

./test_vgae_np_classification_node.sh

./test_vgae_np_classification_operator.sh

echo "All scripts have completed."
