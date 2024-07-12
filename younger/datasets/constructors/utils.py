#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 22:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import math

from younger.datasets.utils.constants import READMEPattern


def convert_bytes(size_in_bytes: int) -> str:
    if size_in_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_in_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_in_bytes / p, 2)
    return f'{s} {size_name[i]}'


def get_instance_name_parts(instance_name: str) -> tuple[str, str, str]:
    model_name, instance_name_left = tuple(instance_name.split('--MN_YD_MS--'))
    model_source, onnx_model_filestem = tuple(instance_name_left.split('--MS_YD_ON--'))
    return (model_name, model_source, onnx_model_filestem)


def get_instance_dirname(model_name: str, model_source: str, onnx_model_filestem: str) -> str:
    return model_name + '--MN_YD_MS--' + model_source + '--MS_YD_ON--' + onnx_model_filestem


def extract_table_related_metrics_from_readme(readme: str) -> list[dict[str, list[str]]]:

    def extract_cells(row: str) -> list[str]:
        cell_str = row.strip()
        cell_str = cell_str[ 1:  ] if len(cell_str) and cell_str[ 0] == '|' else cell_str
        cell_str = cell_str[  :-1] if len(cell_str) and cell_str[-1] == '|' else cell_str
        cells = [cell.strip() for cell in cell_str.split('|')]
        return cells

    readme = readme.strip() + '\n'
    table_related = list()
    for match_result in re.finditer(READMEPattern.TABLE, readme, re.MULTILINE):
        headers = match_result.group(1)
        headers = extract_cells(headers)
        rows = list()
        for row in match_result.group(3).strip().split('\n'):
            rows.append(extract_cells(row))
        table_related.append(
            dict(
                headers=headers,
                rows=rows
            )
        )

    return table_related


def extract_digit_related_metrics_from_readme(readme: str) -> list[str]:

    def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        intervals = sorted(intervals)
        new_intervals = list()
        start, end = (-1, -1)
        for interval in intervals:
            if end < interval[0]:
                new_interval = (start, end)
                new_intervals.append(new_interval)
                start, end = interval

            else:
                if end < interval[1]:
                    end = interval[1]

        new_intervals.append((start, end))
        return new_intervals[1:]

    intervals = list()
    for match_result in re.finditer(READMEPattern.DIGIT, readme, re.MULTILINE):
        start = match_result.start() - 32
        end = match_result.end() + 32
        intervals.append((start, end))

    intervals = merge_intervals(intervals)
    digit_related = list()
    for start, end in intervals:
        digit_context = ' '.join(readme[start:end].split())
        digit_related.append(digit_context)

    return digit_related


def filter_readme_filepaths(filepaths: list[str]) -> list[str]:
    readme_filepaths = list()
    pattern = re.compile(r'.*readme(?:\.[^/\\]*)?$', re.IGNORECASE)
    for filepath in filepaths:
        if re.match(pattern, filepath) is not None:
            readme_filepaths.append(filepath)

    return readme_filepaths