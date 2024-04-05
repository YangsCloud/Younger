#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 22:07
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import re
import pathlib
import requests
import itertools

from typing import Any, Literal, Iterable, Generator
from huggingface_hub import utils, HfFileSystem, ModelCard, ModelCardData, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from yaml.scanner import ScannerError

from younger.datasets.utils.io import delete_dir
from younger.datasets.utils.logging import logger
from younger.datasets.utils.constants import READMEPattern

from younger.datasets.constructors.utils import extract_table_related_metrics_from_readme, extract_digit_related_metrics_from_readme


def huggingface_paginate(huggingface_path: str, params: dict, headers: dict) -> Iterable[dict]:
    # [NOTE] The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_pagination.py)
    # paginate is called by huggingface_hub.HfApi.list_models();

    session = requests.Session()
    response = session.get(huggingface_path, params=params, headers=headers)
    yield from response.json()

    # Follow pages
    # Next link already contains query params
    next_page_path = response.links.get("next", {}).get("url")
    while next_page_path is not None:
        response = session.get(next_page_path, headers=headers)
        yield from response.json()
        next_page_path = response.links.get("next", {}).get("url")


def get_huggingface_model_infos(
    filter_list: list[str] | None = None,
    full: bool | None = None,
    limit: int | None = None,
    config: bool | None = None,
    sort: str | None = None,
    direction: Literal[-1] = None,
    token: str = None,
) -> Iterable[dict]:
    # [NOTE] The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py [Method: list_models])

    huggingface_path = 'https://huggingface.co/api/models'
    params = dict()
    # The filter_list is should be a tuple that contain multiple str, each str can be a identifier about library, language, task, tags, but not model_name and author.
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

    huggingface_model_infos = huggingface_paginate(huggingface_path, params=params, headers=headers)

    if limit is not None:
        huggingface_model_infos = itertools.islice(huggingface_model_infos, limit)
    return huggingface_model_infos


def get_huggingface_model_readmes(model_ids: list[str]) -> Generator[str, None, None]:
    hf_file_system = HfFileSystem()
    for model_id in model_ids:
        if hf_file_system.exists(f'{model_id}/README.md'):
            try:
                with hf_file_system.open(f'{model_id}/README.md', mode='r', encoding='utf-8') as readme_file:
                    readme = readme_file.read()
                    readme = readme.replace('\t', ' ')
                    yield readme
            except UnicodeDecodeError as error:
                logger.error(f"REPO: {model_id}. Encoding Error - The Encoding [UTF-8] are Invalid. - Error: {error}")
                raise error
            except Exception as error:
                logger.error(f"REPO: {model_id}. Encounter An Error {error}.")
                raise error
        else:
            logger.info(f"REPO: {model_id}. No README.md, skip.")
            continue


def remove_card_related_from_readme(readme: str) -> str:
    readme_lines = readme.split('\n')
    split_pattern = '---'
    if len(readme_lines) <= 2:
        return '\n'.join(readme_lines)
    if readme_lines[0].strip() == split_pattern:
        for index, readme_line in enumerate(readme_lines[1:], start=1):
            if readme_line.strip() == split_pattern:
                break
        return '\n'.join(readme_lines[index+1:])
    return '\n'.join(readme_lines)


def extract_candidate_metrics_from_readme(readme: str) -> dict[str, dict[str, Any]]:
    candidate_metrics = dict()

    card_related = dict()
    try:
        card_data: ModelCardData = ModelCard(readme, ignore_metadata_errors=True).data
    except ScannerError as error:
        logger.error(f' !!! Return Empty Card !!! Format of YAML at the Begin of README File Maybe Wrong. Error: {error}')
        raise error
    except ValueError as error:
        logger.error(f' !!! YAML ValueError !!! Format of YAML at the Begin of README File Maybe Wrong. Error: {error}')
        raise error
    except Exception as error:
        logger.error(f' !!! Unknow ModelCard Parse Error !!! Format of YAML at the Begin of README File Maybe Wrong. Error: {error}')
        raise error

    card_related['datasets'] = card_data.datasets if card_data.datasets else list()
    card_related['metrics'] = card_data.metrics if card_data.metrics else list()

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
    card_related['results'] = results

    candidate_metrics['card_related'] = card_related

    candidate_metrics['table_related'] = extract_table_related_metrics_from_readme(readme)

    readme = re.sub(READMEPattern.TABLE, '', readme)
    readme = re.sub(READMEPattern.DATE, '', readme)
    readme = re.sub(READMEPattern.DATETIME, '', readme)

    candidate_metrics['digit_related'] = extract_digit_related_metrics_from_readme(readme)
    
    return candidate_metrics


def infer_model_size(model_id: str) -> int:
    hf_file_system = HfFileSystem()
    filenames = list()
    filenames.extend(hf_file_system.glob(model_id+'/*.bin'))
    filenames.extend(hf_file_system.glob(model_id+'/*.h5'))
    filenames.extend(hf_file_system.glob(model_id+'/*.ckpt'))
    filenames.extend(hf_file_system.glob(model_id+'/*.msgpack'))
    filenames.extend(hf_file_system.glob(model_id+'/*.safetensors'))
    infered_model_size = 0
    for filename in filenames:
        meta_data = get_hf_file_metadata(hf_hub_url(repo_id=model_id, filename=filename[len(model_id)+1:]))
        infered_model_size += meta_data.size
    return infered_model_size


def clean_cache_root(cache_dirpath: pathlib.Path):
    info = scan_cache_dir(cache_dirpath)
    commit_hashes = list()
    for repo in list(info.repos):
        for revision in list(repo.revisions):
            commit_hashes.append(revision.commit_hash)
    delete_strategy = info.delete_revisions(*commit_hashes)
    delete_strategy.execute()


def clean_default_cache_repo(repo_id: str):
    repo_id = repo_id.replace("/", "--")
    repo_type = "model"

    default_cache_dirpath = pathlib.Path(HUGGINGFACE_HUB_CACHE)
    repo_cache_dirpath = default_cache_dirpath.joinpath(f"{repo_type}s--{repo_id}")

    if default_cache_dirpath.is_dir():
        delete_dir(repo_cache_dirpath)


def clean_specify_cache_repo(repo_id: str, specify_cache_dirpath: pathlib.Path):
    repo_id = repo_id.replace("/", "--")
    repo_type = "model"

    repo_cache_dirpath = specify_cache_dirpath.joinpath(f"{repo_type}s--{repo_id}")

    if repo_cache_dirpath.is_dir():
        delete_dir(repo_cache_dirpath)