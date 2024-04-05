#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-05 09:41
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pandas
import pathlib

from huggingface_hub import list_metrics


def get_huggingface_metrics(save_dirpath: pathlib.Path):
    metrics = list_metrics()
    ids = [metric.id for metric in metrics]
    descriptions = [metric.description for metric in metrics]
    data_frame = pandas.DataFrame({'Metric Names': ids, 'Descriptions': descriptions})
    save_filepath = save_dirpath.joinpath('huggingface_metrics.xlsx')
    data_frame.to_excel(save_filepath, index=False)


def get_huggingface_models(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path):
    pass