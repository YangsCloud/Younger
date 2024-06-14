#!/bin/bash

./train_gae_np_encoder_node.sh

./train_gae_np_encoder_operator.sh

./train_vgae_np_encoder_operator.sh

./train_vgae_np_encoder_node.sh

echo "All scripts have completed."