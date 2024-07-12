#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-07-12 09:27
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import tqdm
import pathlib
import requests
import multiprocessing

from typing import Generator, Literal
from younger.commons.io import tar_archive, create_dir, delete_dir
from younger.commons.hash import hash_string
from younger.commons.logging import logger

from younger.datasets.modules import Instance
from younger.datasets.utils.constants import YoungerAPI
from younger.datasets.constructors.utils import get_instance_name_parts
from younger.datasets.constructors.official.api.schema import SeriesCompleteItem


FILES_PREFIX = YoungerAPI.API_ADDRESS + 'files'
SERIES_COMPLETE_PREFIX = YoungerAPI.API_ADDRESS + YoungerAPI.SERIES_COMPLETE_POINT


def get_headers(token: str):
    return {
        "Authorization": f"Bearer {token}"
    }


def read_series_complete_count(token: str) -> int:
    headers = get_headers(token)
    response = requests.get(SERIES_COMPLETE_PREFIX + '?aggregate[count]=*', headers=headers)
    data = response.json()
    return data['data'][0]['count']


def read_series_complete_items(token: str, limit: int = 100) -> Generator[str, None, None]:
    headers = get_headers(token)

    count = read_series_complete_count(token)
    logger.info(f'Find Total {count} Instances')

    limit = 100
    quotient, remainder = divmod(count, limit)
    pages = quotient + (remainder > 0)

    with tqdm.tqdm(total=count, desc='Retrieving') as progress_bar:
        for page in range(1, pages+1):
            response = requests.get(SERIES_COMPLETE_PREFIX + f'?limit={limit}&page={page}&fields[]=instance_name&sort[]=id', headers=headers)
            data = response.json()
            for item in data['data']:
                yield item['instance_name']
                progress_bar.update(1)


def create_series_complete_item(series_complete_item: SeriesCompleteItem, token: str) -> tuple[str, str]:
    headers = get_headers(token)
    response = requests.post(SERIES_COMPLETE_PREFIX, headers=headers, json=[series_complete_item.dict()])
    data = response.json()
    if 'data' in data:
        flag = (data['data'][0]['instance_name'], 'succ')
    else:
        flag = (f'{series_complete_item.instance_name}: {str(data)}', 'fail')
    return flag


def generate_instance_meta(instance_dirpath: pathlib.Path, meta_filepath: pathlib.Path, save: bool = False) -> dict:
    instance_meta = dict()
    instance = Instance()
    instance.load(instance_dirpath)

    instance_meta['node_number'] = instance.network.graph.number_of_nodes()
    instance_meta['edge_number'] = instance.network.graph.number_of_edges()

    # TODO: Save more statistics in to the META file.
    if save:
        pass

    return instance_meta


def upload_instance(parameter: tuple[pathlib.Path, pathlib.Path, str]):
    (instance_dirpath, cache_dirpath, since_version, paper, token) = parameter

    instance_filename = hash_string(instance_dirpath.name + f'-Paper:{paper}', hash_algorithm='blake2b', digest_size=16)
    # instance_filename = instance_dirpath.name

    meta_filepath = cache_dirpath.joinpath(instance_filename + '.json')
    instance_meta = generate_instance_meta(instance_dirpath, meta_filepath)

    archive_filepath = cache_dirpath.joinpath(instance_filename + '.tgz')
    tar_archive(instance_dirpath, archive_filepath, compress=True)

    model_name, model_source, model_part = get_instance_name_parts(instance_dirpath.name)
    if model_source == 'HuggingFace':
        model_name = model_name.replace('--HF--', '/')
        model_source = model_source.lower()
    elif model_source == 'ONNX':
        model_name = model_name.replace('--TV--', '/')
        model_source = model_source.lower()
    elif model_source == 'TorchVision':
        model_name = model_name.replace('--TV--', '/')
        model_source = 'pytorch'
    else:
        raise ValueError('Not A Valid Directory Path Name.')

    headers = get_headers(token)

    with open(archive_filepath, 'rb') as archive_file:
        payload = dict()
        payload['storage'] = 'vultr'
        payload['folder'] = '6ba43f7f-8cf1-49bb-894f-0ac75e3b5b0f'
        payload['title'] = archive_filepath.name
        files = (
            ('file', (archive_filepath.name, archive_file, 'application/gzip')),
        )
        response = requests.post(FILES_PREFIX, headers=headers, data=payload, files=files)
        try:
            data = response.json()
        except:
            print(response)
            print(response.text)
            import sys
            sys.exit(1)
        instance_tgz_id = data['data']['id']

    series_complete_item = SeriesCompleteItem(
        instance_name=instance_filename,
        model_name=model_name,
        model_source=model_source,
        model_part=model_part,
        node_number=instance_meta['node_number'],
        edge_number=instance_meta['edge_number'],
        since_version=since_version,
        paper=paper,
        status='access',
        instance_tgz=instance_tgz_id
    )

    flag = create_series_complete_item(series_complete_item=series_complete_item, token=token)
    return flag


def main(dataset_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, worker_number: int = 4, since_version: str = '0.0.0', paper: bool = False, token: str = None):
    logger.info(f'Checking Cache Directory Path: {cache_dirpath}')
    if cache_dirpath.is_dir():
        cache_content = [path for path in cache_dirpath.iterdir()]
        assert len(cache_content) == 0, 'You Need Specify An Empty Cache Directory.'
    else:
        create_dir(cache_dirpath)

    logger.info(f'Retrieving Already Inserted Instances...')
    exist_instances = list(read_series_complete_items(token))
    logger.info(f'Retrieved Total {len(exist_instances)}.')

    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters: list[tuple[pathlib.Path, pathlib.Path]] = list()
    for path in dataset_dirpath.iterdir():
        if path.is_dir():
            if path.name not in exist_instances:
                parameters.append((path, cache_dirpath, since_version, paper, token))

    logger.info(f'Total Instances To Be Uploaded: {len(parameters)}')

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Uploading') as progress_bar:
            for index, flag in enumerate(pool.imap_unordered(upload_instance, parameters), start=1):
                if flag[1] == 'fail':
                    logger.error(f'FAIL: {flag[0]}')
                    break
                progress_bar.update(1)
    delete_dir(cache_dirpath, only_clean=True)