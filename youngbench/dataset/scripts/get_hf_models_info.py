#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-10 15:24
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pathlib
import requests
import argparse
import itertools

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
PROXIES = {'https': 'http://127.0.0.1:12345'}
TOKEN = None

from typing import Dict, List, Literal, Iterable, Optional
from huggingface_hub import utils, snapshot_download, login

from youngbench.logging import set_logger, logger


def paginate(hf_path: str, params: Dict, headers: Dict) -> Iterable[Dict]:
    # [NOTE] The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/f386b2ae74bf18443836936941ae8bd1bfd40903/src/huggingface_hub/utils/endpoint_helpers.py#L156)
    # paginate is called by huggingface_hub.HfApi.list_models();

    session = requests.Session()
    response = session.get(hf_path, params=params, headers=headers)
    yield from response.json()

    # Follow pages
    # Next link already contains query params
    next_page_path = response.links.get("next", {}).get("url")
    while next_page_path is not None:
        response = session.get(next_page_path, headers=headers)
        yield from response.json()
        next_page_path = response.links.get("next", {}).get("url")


def get_model_infos(
    filter_list: Optional[List[str]] = None,
    full: Optional[bool] = None,
    limit: Optional[int] = None,
    config: Optional[bool] = None,
    sort: Optional[str] = None,
    direction: Optional[Literal[-1]] = None,
) -> Iterable[Dict]:
    # [NOTE] The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L1378)

    hf_path = 'https://huggingface.co/api/models'
    params = dict()
    # filter_list is should be a tuple that contain multiple str, each str can be a identifier about library, language, task, tags, but not model_name and author.
    # See details at https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L1514
    if filter_list is not None:
        params['filter'] = tuple(filter_list)
    # Whether to fetch all model data, including the `last_modified`, the `sha`, the files and the `tags`.
    if full is not None:
        params['full'] = full
    # The limit on the number of models fetched. Leaving this option to `None` fetches all models.
    if limit is not None:
        params['limit'] = limit
    # Whether to fetch the model configs as well. This is not included in `full` due to its size.
    if config is not None:
        params['config'] = config
    # The key with which to sort the resulting models. Possible values are the str values of the filter_list.
    if sort is not None:
        params['sort'] = sort
    # Direction in which to sort. The value `-1` sorts by descending order while all other values sort by ascending order.
    if direction is not None:
        params['direction'] = direction

    headers = utils.build_hf_headers(token=TOKEN)

    model_infos = paginate(hf_path, params=params, headers=headers)

    if limit is not None:
        model_infos = itertools.islice(model_infos, limit)
    return model_infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LoadCreate/Update The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Model Info Dir
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--sort', type=bool, default=False)
    parser.add_argument('--sortby', type=str, default=None)
    parser.add_argument('--ascend', type=bool, default=False)
    parser.add_argument('--filter', type=str, default=['onnx'], nargs='+')
    parser.add_argument('--save-path', type=str, default='./model_infos.json')
    parser.add_argument('--logging-path', type=str, default='./get_model_infos.log')

    args = parser.parse_args()

    set_logger(path=args.logging_path)

    assert 0 < args.num

    if args.sort:
        sort_key = args.sortby
        direction = args.ascend - 1
    else:
        sort_key = None
        direction = None

    login(token=TOKEN)

    save_path = pathlib.Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f' = Fetching {args.num} Models\' Info (Sort={args.sort}, Ascend={args.ascend}) ... ')
    model_infos = get_model_infos(filter_list=args.filter, full=True, limit=args.num, config=True, sort=sort_key, direction=direction)
    with open(save_path, 'w') as f:
        model_infos = list(model_infos)
        json.dump(model_infos, f, indent=2)
    logger.info(f' - Models\' Info are saved into: {save_path.absolute()}')