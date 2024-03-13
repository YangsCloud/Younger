#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-11 20:59
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import pathlib
import tempfile
import argparse

from typing import Any
from yoolkit import text
from huggingface_hub import login, HfFileSystem, hf_hub_download

from youngbench.logging import set_logger, logger
from youngbench.dataset.construct.utils.schema import HFInfo, Model
from youngbench.dataset.construct.utils.action import create_hfinfo_item, create_hfinfo_items, update_model_item_by_model_id, update_model_items_by_model_ids, read_model_items_manually, read_hfinfo_items_manually, read_limit_model_items

temp_dir = tempfile.mkdtemp()
table_pattern = r'(\|?(?:[^\r\n\|]*\|)+(?:[^\r\n]*\|?))\r?\n(\|?(?:(?:\s*:?-+:?\s*)\|)+(?:(?:\s*:?-+:?\s*)\|?))\r?\n((?:\|?(?:(?:[^\r\n\|]*)\|)+(?:(?:(?:[^\r\n\|]*)\|?))\r?\n)+)'
# digit_pattern = r'(?:[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))%?)'
digit_pattern = r'(?:[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))%?)\s+|\s+(?:[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))%?)'
date_pattern = r'(?:(?:\d{4})(?:-|\/)(?:\d{1,2})(?:-|\/)\d{1,2})|(?:(?:\d{1,2})(?:-|\/)(?:\d{1,2})(?:-|\/)\d{4})|(?:(?:\d{4})(?:-|\/)(?:\d{1,2}))|(?:(?:\d{1,2})(?:-|\/)(?:\d{4}))|(?:\d{1,2}(?:-|\/)\d{1,2})'
datetime_pattern = r'\b\d{4}-(?!(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b)\d{1,2}-(?!([12]\d|3[01])\b)\d{1,2} \d{1,2}:\d{2}(:\d{2})?\b|\b\d{1,2}:\d{2}(:\d{2})?(?:\s*[apAP]\.?[mM]\.?)?\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'


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


def fetch_metrics(lines: list[str]) -> dict[str, Any]:
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
    digit_relate = list()
    for match_result in re.finditer(digit_pattern, content, re.MULTILINE):
        start = match_result.start() - 32
        end = match_result.end() + 32
        digit_context = ' '.join(content[start:end].split())
        digit_relate.append(digit_context)

    metrics = dict(
        table_relate=table_relate,
        digit_relate=digit_relate,
        clean=dict()
    )
    return metrics


def extract_all_metrics(readme_filepaths: list[str], fs: HfFileSystem, save_dirpath: str | None = None, only_download: bool = False) -> dict[str, dict[str, Any]]:
    save_dirpath = save_dirpath or temp_dir

    all_metrics = dict()
    for readme_filepath in readme_filepaths:
        readme_savepath = pathlib.Path(save_dirpath).joinpath(readme_filepath)
        readme_savepath.parent.mkdir(parents=True, exist_ok=True)
        fs.download(readme_filepath, lpath=str(readme_savepath))
        if only_download:
            continue
        try:
            with open(readme_savepath, encoding='utf-8') as readme:
                all_metrics[readme_filepath] = fetch_metrics(readme.readlines())
        except UnicodeDecodeError as e:
            logger.info(f" Encoding Error - Now Detect The Encoding Mode. - Error: {e}")
            encoding = text.detect_file_encoding(readme_savepath)
            try:
                with open(readme_savepath, encoding=encoding) as readme:
                    all_metrics[readme_filepath] = fetch_metrics(readme.readlines())
            except Exception as e:
                all_metrics[readme_filepath] = dict()
                logger.info(f" Extract Failed. Skip! {readme_filepath} - Error: {e}")
        except Exception as e:
            all_metrics[readme_filepath] = dict()
            logger.info(f" Extract Failed. Skip! {readme_filepath} - Error: {e}")
    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enrich The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    parser.add_argument('--token', type=str, required=True)

    parser.add_argument('--number', type=int, default=1)

    parser.add_argument('--report', type=int, default=100)

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--save-dirpath', type=str, default=None)

    parser.add_argument('--logging-path', type=str, default=None)

    parser.add_argument('--only-download', action='store_true')

    args = parser.parse_args()

    assert args.number > 0

    assert args.report > 0

    if args.hf_token is not None:
        login(token=args.hf_token)

    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    fs = HfFileSystem()

    logger.info(f' = Fetching *README* & Filling *Metrics* For All Models ...')

    # YBDM.Maintaining = False: Basic Fetched;
    # YBDM.Maintaining = True: README Checked;
    # All Items Must Be Inserted Into HFI **BEFORE** *README* and *Metrics* Processed;
    # Items in HFI Can Be Downloaded Then;
    index = 0
    success = 0
    failure = 0
    filter = {
        'maintaining': {
            '_eq': 'false'
        },
        'model_source': {
            '_eq': 'HuggingFace'
        }
    }
    batch = list()
    models = read_limit_model_items(args.token, limit=args.number, fields=['model_id'], filter=filter)
    while models:
        if models:
            for model in models:
                index += 1
                filepaths = fs.glob(f'{model.model_id}/**')
                all_metrics = extract_all_metrics(filter_readme_filepaths(filepaths), fs, save_dirpath=args.save_dirpath, only_download=args.only_download)
                if args.only_download:
                    if index % args.report == 0:
                        logger.info(f' - Only Download - [Index: {index}]')
                    continue
                new_model = Model(all_metrics=all_metrics, maintaining=True)

                if len(batch) < args.number:
                    batch.append((new_model, model.model_id))

                if len(batch) == args.number:
                    for new_model, model_id in batch:
                        result = update_model_item_by_model_id(model_id, new_model, args.token)
                        if result is None:
                            failure += 1
                            logger.info(f' - No.{index} Item Creation Error - Model ID: {model_id}')
                        else:
                            success += 1
                    batch = list()

                if index % args.report == 0:
                    logger.info(f' - [Index: {index}] Success/Failure/OnRoud:{success}/{failure}/{len(batch)}')
        models = read_limit_model_items(args.token, limit=args.number, fields=['model_id'], filter=filter)

    if not args.only_download:
        if len(batch) > 0:
            for new_model, model_id in batch:
                result = update_model_item_by_model_id(model_id, new_model, args.token)
                if result is None:
                    failure += 1
                    logger.info(f' - No.{index} Item Creation Error - Model ID: {model_id}')
                else:
                    success += 1
            batch = list()
        logger.info(f' = END [Index: {index}] Success/Failure/OnRoud:{success}/{failure}/{len(batch)}')
    else:
        logger.info(f' = END = Only Download - [Index: {index}]')