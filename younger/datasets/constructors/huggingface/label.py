#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-12 00:26
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from younger.datasets.utils.metrics.metric_parser import parse_metric


def get_task_label(task_string: str) -> str | None:
    pass


def get_dataset_label(dataset_string: str) -> str | None:
    pass


def get_metric_name_string(hf_metric_type: str, hf_metric_name: str | None = None, hf_metric_class: str | None = None) -> str:
    hf_metric_type = hf_metric_type.lower()
    hf_metric_name = hf_metric_name.lower() if hf_metric_name else None
    if hf_metric_name is None:
        metric_name_string = hf_metric_type
    else:
        if len(hf_metric_type) < len(hf_metric_name):
            metric_name_string = hf_metric_name
        else:
            metric_name_string = hf_metric_type

    return metric_name_string


def get_metric_label(metric_name_string: str, metric_value_string: str) -> tuple[str, float] | None:
    return parse_metric(metric_name_string, metric_value_string)