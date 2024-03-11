#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-11 20:08
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import requests
import itertools

from typing import Dict, List, Literal, Iterable, Optional
from huggingface_hub import utils, login


def hf_paginate(hf_path: str, params: Dict, headers: Dict) -> Iterable[Dict]:
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


def get_hf_model_infos(
    filter_list: Optional[List[str]] = None,
    full: Optional[bool] = None,
    limit: Optional[int] = None,
    config: Optional[bool] = None,
    sort: Optional[str] = None,
    direction: Optional[Literal[-1]] = None,
    token: Optional[str] = None,
) -> Iterable[Dict]:
    # [NOTE] The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L1378)

    hf_path = 'https://huggingface.co/api/models'
    params = dict()
    # filter_list is should be a tuple that contain multiple str, each str can be a identifier about library, language, task, tags, but not model_name and author.
    # See details at https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L1514
    if filter_list is not None:
        params['filter'] = tuple(filter_list)
    # The limit on the number of models fetched. Leaving this option to `None` fetches all models.
    if limit is not None:
        params['limit'] = limit
    # The key with which to sort the resulting models. Possible values are the str values of the filter_list.
    if sort is not None:
        params['sort'] = sort
    # Direction in which to sort. The value `-1` sorts by descending order while all other values sort by ascending order.
    if direction is not None:
        params['direction'] = direction

    # Whether to fetch all model data, including the `last_modified`, the `sha`, the files and the `tags`.
    if full:
        params['full'] = full
    # Whether to fetch the model configs as well. This is not included in `full` due to its size.
    if config:
        params['config'] = config

    headers = utils.build_hf_headers(token=token)

    hf_model_infos = hf_paginate(hf_path, params=params, headers=headers)

    if limit is not None:
        hf_model_infos = itertools.islice(hf_model_infos, limit)
    return hf_model_infos


def get_hf_model_readmes(

):
    pass