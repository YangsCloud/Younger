#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-18 08:12
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas
import argparse

from huggingface_hub import list_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get All HuggingFace Metrics.")
    
    parser.add_argument('--save-filepath', type=str, default='./metrics.xlsx')

    args = parser.parse_args()

    metrics = list_metrics()
    ids = [metric.id for metric in metrics]
    descriptions = [metric.description for metric in metrics]
    data_frame = pandas.DataFrame({'Metric Names': ids, 'Descriptions': descriptions})
    data_frame.to_excel(args.save_filepath, index=False)