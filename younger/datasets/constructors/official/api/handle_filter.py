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
import json
import pathlib
import requests
import multiprocessing

from typing import Generator, Literal
from younger.commons.io import tar_archive, create_dir, delete_dir, save_json
from younger.commons.hash import hash_string
from younger.commons.logging import logger

from younger.datasets.modules import Instance
from younger.datasets.utils.constants import YoungerAPI
from younger.datasets.constructors.utils import get_instance_name_parts
from younger.datasets.constructors.official.api.schema import SeriesFilterItem


FILES_PREFIX = YoungerAPI.API_ADDRESS + 'files'
SERIES_FILTER_PREFIX = YoungerAPI.API_ADDRESS + YoungerAPI.SERIES_FILTER_POINT


def get_headers(token: str):
    return {
        "Authorization": f"Bearer {token}"
    }


def read_series_filter_count(token: str) -> int:
    headers = get_headers(token)
    response = requests.get(SERIES_FILTER_PREFIX + '?aggregate[count]=*', headers=headers)
    data = response.json()
    return data['data'][0]['count']


def read_series_filter_items(token: str, limit: int = 100) -> Generator[str, None, None]:
    headers = get_headers(token)

    count = read_series_filter_count(token)
    logger.info(f'Find Total {count} Instances')

    limit = 100
    quotient, remainder = divmod(count, limit)
    pages = quotient + (remainder > 0)

    with tqdm.tqdm(total=count, desc='Retrieving') as progress_bar:
        for page in range(1, pages+1):
            response = requests.get(SERIES_FILTER_PREFIX + f'?limit={limit}&page={page}&fields[]=instance_hash&sort[]=id', headers=headers)
            data = response.json()
            for item in data['data']:
                yield item['instance_hash']
                progress_bar.update(1)


def create_series_filter_item(series_filter_item: SeriesFilterItem, token: str) -> tuple[str, str]:
    headers = get_headers(token)
    response = requests.post(SERIES_FILTER_PREFIX, headers=headers, json=[series_filter_item.dict()])
    data = response.json()
    if 'data' in data:
        flag = (data['data'][0]['instance_hash'], 'succ')
    else:
        flag = (f'{series_filter_item.instance_hash}: {str(data)}', 'fail')
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
    (instance_dirpath, cache_dirpath, meta, with_attributes, since_version, paper, token) = parameter

    instance_filename = hash_string(instance_dirpath.name + f'-Paper:{paper}-With_Attributes:{with_attributes}', hash_algorithm='blake2b', digest_size=16)
    # instance_filename = instance_dirpath.name

    meta_filepath = cache_dirpath.joinpath(instance_filename + '.json')
    instance_meta = generate_instance_meta(instance_dirpath, meta_filepath, save=meta)

    archive_filepath = cache_dirpath.joinpath(instance_filename + '.tgz')
    tar_archive(instance_dirpath, archive_filepath, compress=True)

    headers = get_headers(token)

    if meta:
        with open(meta_filepath, 'rb') as meta_file:
            payload = dict()
            payload['storage'] = 'vultr'
            payload['folder'] = 'ce8c9263-e584-409b-bd99-ebf2453c6d38'
            payload['title'] = meta_filepath.name
            files = (
                ('file', (meta_filepath.name, meta_file, 'application/json')),
            )
            response = requests.post(FILES_PREFIX, headers=headers, data=payload, files=files)
            data = response.json()
            instance_meta_id = data['id']

    with open(archive_filepath, 'rb') as archive_file:
        payload = dict()
        payload['storage'] = 'vultr'
        payload['folder'] = 'ce8c9263-e584-409b-bd99-ebf2453c6d38'
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

    series_filter_item = SeriesFilterItem(
        instance_hash=instance_filename,
        node_number=instance_meta['node_number'],
        edge_number=instance_meta['edge_number'],
        with_attributes=with_attributes,
        since_version=since_version,
        paper=paper,
        status='access',
        instance_meta=instance_meta_id if meta else None,
        instance_tgz=instance_tgz_id
    )

    flag = create_series_filter_item(series_filter_item=series_filter_item, token=token)
    return flag


def main(dataset_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, worker_number: int = 4, meta: bool = False, with_attributes: bool = False, since_version: str = '0.0.0', paper: bool = False, token: str = None):
    logger.info(f'Checking Cache Directory Path: {cache_dirpath}')
    if cache_dirpath.is_dir():
        cache_content = [path for path in cache_dirpath.iterdir()]
        assert len(cache_content) == 0, 'You Need Specify An Empty Cache Directory.'
    else:
        create_dir(cache_dirpath)

    logger.info(f'Retrieving Already Inserted Instances...')
    exist_instances = list(read_series_filter_items(token))
    logger.info(f'Retrieved Total {len(exist_instances)}.')

    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters: list[tuple[pathlib.Path, pathlib.Path]] = list()
    for path in dataset_dirpath.iterdir():
        if path.is_dir():
            if path.name not in exist_instances:
                parameters.append((path, cache_dirpath, meta, with_attributes, since_version, paper, token))

    logger.info(f'Total Instances To Be Uploaded: {len(parameters)}')

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Uploading') as progress_bar:
            for index, flag in enumerate(pool.imap_unordered(upload_instance, parameters), start=1):
                if flag[1] == 'fail':
                    logger.error(f'FAIL: {flag[0]}')
                    break
                progress_bar.update(1)
    
    delete_dir(cache_dirpath, only_clean=True)