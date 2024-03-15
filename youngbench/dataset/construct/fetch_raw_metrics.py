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

import json
import pathlib
import argparse

from youngbench.logging import set_logger, logger
from youngbench.dataset.construct.utils.action import get_models_count, read_model_items_manually

CARD_RESULT_LABEL = ["metric_type", "metric_value", "metric_config", "task_type", "dataset_type", "dataset_config", "dataset_split"]
CONCAT_CHAR = ' =|= '
NULL = '_NULL_'


def split(raw_metrics) -> tuple:
    cards = raw_metrics['cards_relate']
    table = raw_metrics['table_relate']
    digit = raw_metrics['digit_relate']

    cards_datasets_str = CONCAT_CHAR.join(cards['datasets'])
    cards_metrics_str = CONCAT_CHAR.join(cards['metrics'])
    cards_results_strs = list()
    for result in cards['results']:
        cards_result = list()
        for key in CARD_RESULT_LABEL:
            cards_result.append(f'< {key} := {result.get(key, NULL)} >')
        cards_results_str = CONCAT_CHAR.join(cards_result)
        cards_results_strs.append(cards_results_str)
    
    table_strs = list()
    for tab in table:
        good_headers = list()
        for index, cell in enumerate(tab['headers']):
            good_headers.append(f'[.{index}.] {cell}')
        tab_headers_str = CONCAT_CHAR.join(good_headers)
        tab_rows_strs = list()
        for row in tab['rows']:
            good_row = list()
            for index, cell in enumerate(row):
                good_row.append(f'[.{index}.] {cell}')
            tab_rows_str = CONCAT_CHAR.join(good_row)
            tab_rows_strs.append(tab_rows_str)
        table_strs.append((tab_headers_str, tab_rows_strs))

    return dict(
        cards_datasets_str=cards_datasets_str,
        cards_metrics_str=cards_metrics_str,
        cards_results_strs=cards_results_strs,
        table_strs=table_strs,
        digit=digit
    )


def check(splits: dict) -> bool:
    for _, split in splits.items():
        if len(split) != 0:
            return True
    return False


def pretty(splits: dict) -> str:
    pretty_card_results_str = ''
    for index, cards_results_str in enumerate(splits['cards_results_strs']):
        pretty_card_results_str = pretty_card_results_str + f'    No. {index} - {cards_results_str}\n'
    if len(pretty_card_results_str) == 0:
        pretty_card_results_str = '    // No Information Provided!\n'

    pretty_table_str = ''
    for index, table_str in enumerate(splits['table_strs']):
        pretty_table_str = pretty_table_str + f'    No. {index} Table:\n'
        pretty_table_str = pretty_table_str + f'      Headers - {table_str[0]}\n'
        pretty_table_str = pretty_table_str + f'        ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\n'
        for row_index, row_str in enumerate(table_str[1]):
            pretty_table_str = pretty_table_str + f'      Row {row_index} - {table_str[1][row_index]}\n'
        pretty_table_str = pretty_table_str + f'\n'
    if len(pretty_table_str) == 0:
        pretty_table_str = '    // No Information Provided!\n\n'

    pretty_digit_str = ''
    for index, digit_str in enumerate(splits['digit']):
        pretty_digit_str = pretty_digit_str + f'    No. {index} - {digit_str}\n'
        pretty_digit_str = pretty_digit_str + f'\n'
    if len(pretty_digit_str) == 0:
        pretty_digit_str = '    // No Information Provided!\n'

    NIP_STR="\n    // No Information Provided"
    pretty_str = (
        f'【ModelCard】\n'
        f'  NOTE: ModelCard Provides Precise Infomation. Do Not Annotate This Part! Except For Labeling The Metric Class.\n'
        f'\n'
        f'  Datasets:  {splits["cards_datasets_str"] if splits["cards_datasets_str"] else NIP_STR}\n'
        f'\n'
        f'  Metrics:   {splits["cards_metrics_str"] if splits["cards_metrics_str"] else NIP_STR}\n'
        f'\n'
        f'  Results:\n'
        f'{pretty_card_results_str}'
        f'\n'
        f'  ================================================================================================\n'
        f'\n'
        f'【Tables】\n'
        f'  NOTE: Please Neglect Invalid Tables.\n'
        f'\n'
        f'{pretty_table_str}'
        f'  ================================================================================================\n'
        f'\n'
        f'【Digits Related】\n'
        f'  NOTE: !!! Please Label This Part Carefully !!!\n'
        f'\n'
        f'{pretty_digit_str}'
    )
    return pretty_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch All Raw Metrics For Annotate From The Young Neural Network Architecture Dataset (YoungBench - Dataset).")
    parser.add_argument('--token', type=str, required=True)

    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--l-index', type=int, default=1)
    parser.add_argument('--r-index', type=int, default=1)

    parser.add_argument('--save-dirpath', type=str, default=None)
    parser.add_argument('--logging-path', type=str, default=None)
    args = parser.parse_args()


    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    total_models = get_models_count(args.token)
    assert args.step > 0
    assert 1 <= args.l_index and args.l_index <= total_models
    assert 1 <= args.r_index and args.r_index <= total_models
    assert args.l_index <= args.r_index

    save_dirpath = pathlib.Path(args.save_dirpath)
    save_dirpath.mkdir(parents=True, exist_ok=True)

    logger.info("Checking Exist Models... ")

    ls_template = dict(
        data=dict(
            text="",
            model_id=""
        )
    )

    exact_filepaths = list()
    logger.info(f'Retrieving Items From {args.l_index} To {args.r_index} ...')
    for offset in range(args.l_index, args.r_index + 1, args.step):
        l_id = offset
        r_id = offset + args.step - 1
        models = list(read_model_items_manually(args.token, limit=args.step, filter=dict(id=dict(_between=[l_id, r_id])), fields=['id', 'model_id', 'raw_metrics']))
        logger.info(f" Retrieved Total {len(models)} ({models[0].id} ... {models[-1].id}).")
        logger.info(f" Now Check ...")
        skip = 0
        for model in models:
            splits = split(model.raw_metrics)
            if not check(splits):
                logger.info(f" No Need To Be Labeled {model.model_id}...")
                skip += 1
                continue
            ls_template['data']['model_id'] = model.model_id
            ls_template['data']['text'] = pretty(splits)
            save_filepath = save_dirpath.joinpath(f'model_{model.id}.json')
            with open(save_filepath, 'w') as sf:
                json.dump(ls_template, sf)
        logger.info(f" Retrieved/Skip ({len(models) - skip}/{skip}).")

    logger.info(f"Finish Retrive.")