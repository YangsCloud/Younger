#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-12 00:41
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

TOKEN="SECRET_TOKEN"

python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path /home/jason/0.Research/YoungBench/enrich.log --number 50 --report 200 --tag timm --token ${TOKEN};
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path /home/jason/0.Research/YoungBench/enrich.log --number 50 --report 200 --tag sentence-transformers --token ${TOKEN};
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path /home/jason/0.Research/YoungBench/enrich.log --number 50 --report 200 --tag onnx --token ${TOKEN};
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path /home/jason/0.Research/YoungBench/enrich.log --number 50 --report 200 --tag diffusers --token ${TOKEN};
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path /home/jason/0.Research/YoungBench/enrich.log --number 50 --report 200 --tag transformers --token ${TOKEN};
