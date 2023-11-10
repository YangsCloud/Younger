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

import requests
import argparse
import itertools

from typing import Dict, Literal, Iterable, Optional
from huggingface_hub import utils, snapshot_download


def paginate(hf_path: str, params: Dict, headers: Dict) -> Iterable[Dict]:
    # [NOTE] The Code are modified based on the official Hugging Face Hub source codes.

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
    full: Optional[bool] = None,
    limit: Optional[int] = None,
    config: Optional[bool] = None,
    sort: Optional[str] = None,
    direction: Optional[Literal[-1]] = None,
) -> Iterable[Dict]:

    hf_path = 'https://huggingface.co/api/models'
    params = dict(library='pytorch')
    if full is not None:
        params['full'] = full
    if limit is not None:
        params['limit'] = limit
    if config is not None:
        params['config'] = config
    if sort is not None:
        params['sort'] = sort
    if direction is not None:
        params['direction'] = direction

    headers = utils.build_hf_headers()

    model_infos = paginate(hf_path, params=params, headers=headers)

    if limit is not None:
        model_infos = itertools.islice(model_infos, limit)
    return model_infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create/Update The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Cache Dir
    parser.add_argument('--cache-dir', type=str, default=None)

    args = parser.parse_args()

    top100_downloads_model_infos = get_model_infos(full=True, limit=10, config=True, sort='downloads', direction=-1)

    for index, model_info in enumerate(top100_downloads_model_infos):
        print(f' # {index}: {model_info["id"]}')
        snapshot_download(repo_id=model_info['id'], resume_download=True, cache_dir=args.cache_dir)