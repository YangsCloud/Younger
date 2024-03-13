#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-14 01:13
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import sys
import pathlib
import tempfile

from typing import Any
from yoolkit import text
from huggingface_hub import HfFileSystem, ModelCard, ModelCardData

from youngbench.logging import logger


temp_dir = tempfile.mkdtemp()
table_pattern = r'(\|?(?:[^\r\n\|]*\|)+(?:[^\r\n]*\|?))\r?\n(\|?(?:(?:\s*:?-+:?\s*)\|)+(?:(?:\s*:?-+:?\s*)\|?))\r?\n((?:\|?(?:(?:[^\r\n\|]*)\|)+(?:(?:(?:[^\r\n\|]*)\|?))\r?\n)+)'
# digit_pattern = r'(?:[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))%?)'
digit_pattern = r'(?:[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))%?)\s+|\s+(?:[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))%?)'
date_pattern = r'(?:(?:\d{4})(?:-|\/)(?:\d{1,2})(?:-|\/)\d{1,2})|(?:(?:\d{1,2})(?:-|\/)(?:\d{1,2})(?:-|\/)\d{4})|(?:(?:\d{4})(?:-|\/)(?:\d{1,2}))|(?:(?:\d{1,2})(?:-|\/)(?:\d{4}))|(?:\d{1,2}(?:-|\/)\d{1,2})'
datetime_pattern = r'\b\d{4}-(?!(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b)\d{1,2}-(?!([12]\d|3[01])\b)\d{1,2} \d{1,2}:\d{2}(:\d{2})?\b|\b\d{1,2}:\d{2}(:\d{2})?(?:\s*[apAP]\.?[mM]\.?)?\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'


def realtime_write(info: str):
    sys.stdout.write(info)
    sys.stdout.flush()
    return


def filter_readme_filepaths(filepaths: list[str]) -> list[str]:
    readme_filepaths = list()
    pattern = re.compile(r'.*readme(?:\.[^/\\]*)?$', re.IGNORECASE)
    for filepath in filepaths:
        if re.match(pattern, filepath) is not None:
            readme_filepaths.append(filepath)

    return readme_filepaths


def extract_cells(line: str) -> list[str]:
    cell_str = line.strip()
    cell_str = cell_str[ 1:  ] if cell_str[ 0] == '|' else cell_str
    cell_str = cell_str[  :-1] if cell_str[-1] == '|' else cell_str
    cells = [cell.strip() for cell in cell_str.split('|')]
    return cells


def clean_head(lines: list[str]) -> list[str]:
    split_pattern = '---'
    if len(lines) <= 2:
        return lines
    if lines[0].strip() == split_pattern:
        index = 1
        while lines[index].strip() != split_pattern and index < len(lines):
            index += 1
        return lines[index+1:]


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


def fetch_metrics(lines: list[str]) -> dict[str, Any]:
    lines = clean_head(lines)
    content = ''.join(lines) + '\n'
    table_relate = list()
    for match_result in re.finditer(table_pattern, content, re.MULTILINE):
        headers = match_result.group(1)
        headers = extract_cells(headers)
        rows = list()
        for row in match_result.group(3).strip().split('\n'):
            rows.append(extract_cells(row))
        table_relate.append(
            dict(
                headers=headers,
                rows=rows
            )
        )

    content = re.sub(table_pattern, '', content)
    content = re.sub(date_pattern, '', content)
    content = re.sub(datetime_pattern, '', content)

    intervals = list()
    for match_result in re.finditer(digit_pattern, content, re.MULTILINE):
        start = match_result.start() - 32
        end = match_result.end() + 32
        intervals.append((start, end))

    intervals = merge_intervals(intervals)
    digit_relate = list()
    for start, end in intervals:
        digit_context = ' '.join(content[start:end].split())
        digit_relate.append(digit_context)

    metrics = dict(
        table_relate=table_relate,
        digit_relate=digit_relate,
    )
    return metrics


def fetch_card_relate(readme_filepath: str) -> dict[str, Any]:
    card_relate = dict()
    card_data: ModelCardData = ModelCard.load(readme_filepath, ignore_metadata_errors=True).data
    card_relate['datasets'] = card_data.datasets if card_data.datasets else list()
    card_relate['metrics'] = card_data.metrics if card_data.metrics else list()

    results = list()
    if card_data.eval_results:
        for eval_result in card_data.eval_results:
            result = dict(
                task_type=eval_result.dataset_type,
                dataset_type=eval_result.dataset_type,
                dataset_config=eval_result.dataset_config if eval_result.dataset_config else '',
                dataset_split=eval_result.dataset_split if eval_result.dataset_split else '',
                metric_type=eval_result.metric_type,
                metric_value=eval_result.metric_value,
                metric_config=eval_result.metric_config if eval_result.metric_config else '',
            )
            results.append(result)
    card_relate['results'] = results
    return card_relate


def extract_all_metrics(readme_filepaths: list[str], fs: HfFileSystem, save_dirpath: str | None = None, only_download: bool = False) -> dict[str, dict[str, Any]]:
    save_dirpath = save_dirpath or temp_dir

    all_metrics = dict()
    for readme_filepath in readme_filepaths:
        readme_savepath = pathlib.Path(save_dirpath).joinpath(readme_filepath)
        readme_savepath.parent.mkdir(parents=True, exist_ok=True)
        if not readme_savepath.is_file():
            fs.download(readme_filepath, lpath=str(readme_savepath))

        if only_download:
            continue

        all_metrics[readme_filepath] = dict()
        all_metrics[readme_filepath]['cards_relate'] = fetch_card_relate(readme_savepath)

        try:
            with open(readme_savepath, encoding='utf-8') as readme:
                all_metrics[readme_filepath].update(fetch_metrics(readme.readlines()))
        except UnicodeDecodeError as e:
            logger.info(f" Encoding Error - Now Detect The Encoding Mode. - Error: {e}")
            encoding = text.detect_file_encoding(readme_savepath)
            try:
                with open(readme_savepath, encoding=encoding) as readme:
                    all_metrics[readme_filepath].update(fetch_metrics(readme.readlines()))
            except UnicodeDecodeError as e:
                logger.info(f" Encoding Error - The Encoding [UTF-8 and {encoding}] are Invalid. - Error: {e}")
            except Exception as e:
                logger.info(f" Extract Failed. Skip! {readme_filepath} - Error: {e}")
        except Exception as e:
            logger.info(f" Extract Failed. Skip! {readme_filepath} - Error: {e}")
    return all_metrics