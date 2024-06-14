#!/bin/bash

./train_gae_np_classification_node.sh

./train_gae_np_classification_operator.sh

./train_vgae_np_classification_operator.sh

./train_vgae_np_classification_node.sh

echo "All scripts have completed."