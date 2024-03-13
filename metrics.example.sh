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

python -m youngbench.dataset.construct.fill_metrics_using_hf_readme --token ${TOKEN} --number 1 --report 100 --save-dirpath ~/HF_README --logging-path ~/HF_README/progress.log --only-download;
python -m youngbench.dataset.construct.fill_metrics_using_hf_readme --token ${TOKEN} --number 50 --report 200 --save-dirpath ~/HF_README --logging-path ~/HF_README/progress.log;
