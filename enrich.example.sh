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
HF_TOKEN="HF_TOKEN"

python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path ~/HF_README/enrich.log --token ${TOKEN} --number 50 --report 200 --hf-token ${HF_TOKEN} --save-dirpath ~/HF_README --tag timm ;
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path ~/HF_README/enrich.log --token ${TOKEN} --number 50 --report 200 --hf-token ${HF_TOKEN} --save-dirpath ~/HF_README --tag sentence-transformers ;
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path ~/HF_README/enrich.log --token ${TOKEN} --number 50 --report 200 --hf-token ${HF_TOKEN} --save-dirpath ~/HF_README --tag onnx ;
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path ~/HF_README/enrich.log --token ${TOKEN} --number 50 --report 200 --hf-token ${HF_TOKEN} --save-dirpath ~/HF_README --tag diffusers ;
python -m youngbench.dataset.construct.enrich_dataset_using_hf --logging-path ~/HF_README/enrich.log --token ${TOKEN} --number 50 --report 200 --hf-token ${HF_TOKEN} --save-dirpath ~/HF_README --tag transformers ;
