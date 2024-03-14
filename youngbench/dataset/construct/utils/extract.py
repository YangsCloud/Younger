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
from huggingface_hub import ModelCard, ModelCardData, EvalResult, hf_hub_download, HfFileSystem

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


def filter_readme_filepaths(model_id: str, filepaths: list[str]) -> list[str]:
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
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == split_pattern:
                break
        return lines[index+1:]
    return lines


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


def replace_invalid_chars(filepath: str, logger = logger):
    try:
        with open(filepath, mode='r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError as e:
        logger.info(f" Encoding Error - Now Detect The Encoding Mode. - Error: {e}")
        encoding = text.detect_file_encoding(filepath)
        try:
            with open(filepath, mode='r', encoding=encoding) as file:
                content = file.read()
        except UnicodeDecodeError as e:
            logger.info(f" Encoding Error - The Encoding [UTF-8 and {encoding}] are Invalid. - Error: {e}")
        except Exception as e:
            logger.info(f" Extract Failed. Skip! {filepath} - Error: {e}")
    except Exception as e:
        logger.info(f" Extract Failed. Skip! {filepath} - Error: {e}")
    
    content = content.replace('\t', ' ')

    with open(filepath, mode='w', encoding='utf-8') as file:
        file.write(content)


def fetch_card_relate(readme_filepath: str) -> dict[str, Any]:
    card_relate = dict()
    card_data: ModelCardData = ModelCard.load(readme_filepath, ignore_metadata_errors=True).data
    card_relate['datasets'] = card_data.datasets if card_data.datasets else list()
    card_relate['metrics'] = card_data.metrics if card_data.metrics else list()

    results = list()
    if card_data.eval_results:
        for eval_result in card_data.eval_results:
            result = dict(
                task_type=eval_result.task_type,
                dataset_type=eval_result.dataset_type,
                dataset_config=eval_result.dataset_config if eval_result.dataset_config else '',
                dataset_split=eval_result.dataset_split if eval_result.dataset_split else '',
                metric_type=eval_result.metric_type,
                metric_value=str(eval_result.metric_value),
                metric_config=eval_result.metric_config if eval_result.metric_config else '',
            )
            results.append(result)
    card_relate['results'] = results
    return card_relate


def extract_all_metrics(model_id: str, fs: HfFileSystem, save_dirpath: str | None = None, only_download: bool = False, logger = logger) -> dict[str, dict[str, Any]]:
    save_dirpath = save_dirpath or temp_dir

    all_metrics = dict()
    # for readme_filepath in readme_filepaths:
    save_dirpath: pathlib.Path = pathlib.Path(save_dirpath).joinpath(model_id)
    save_dirpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        readme_filepath = hf_hub_download(model_id, 'README.md', repo_type='model', local_dir=save_dirpath, local_dir_use_symlinks=False)
    except Exception as e:
        if fs.exists(f"{model_id}/README.md"):
            raise e
            # fs.get(f"{model_id}/README.md", readme_filepath)
            # logger.info(f"Downloaded {model_id}/README.md")
        else:
            logger.info(f"REPO: {model_id}. No README.md, Create With Empty Raw_Metrics.")
            return all_metrics

    if only_download:
        return

    replace_invalid_chars(readme_filepath)
    all_metrics['cards_relate'] = fetch_card_relate(readme_filepath)

    with open(readme_filepath, mode='r', encoding='utf-8') as readme:
        all_metrics.update(fetch_metrics(readme.readlines()))
    
    return all_metrics