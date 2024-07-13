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
requests.DEFAULT_RETRIES = 15


def get_headers(token: str):
    return {
        "Authorization": f"Bearer {token}"
    }

 
def read_series_filter_files_count(token: str) -> int:
    headers = get_headers(token)
    response = requests.get(FILES_PREFIX + '?aggregate[count]=*', headers=headers)
    data = response.json()
    return data['data'][0]['count']

def read_series_filter_count(token: str) -> int:
    headers = get_headers(token)
    response = requests.get(SERIES_FILTER_PREFIX + '?aggregate[count]=*', headers=headers)
    data = response.json()
    return data['data'][0]['count']


def delete_series_filter_files(files: dict[str, str], items: list[str], token: str):
    headers = get_headers(token)

    should_be_deleted = set(files.keys()) - set(items)

    with tqdm.tqdm(total=len(should_be_deleted), desc='Deleting') as progress_bar:
        for title in should_be_deleted:
            response = requests.delete(FILES_PREFIX + f'/{files[title]}', headers=headers)
            progress_bar.update(1)


def read_series_filter_files(token: str, limit: int = 100) -> Generator[str, None, None]:
    headers = get_headers(token)

    count = read_series_filter_files_count(token)
    logger.info(f'Find Total {count} Instances')

    limit = 100
    quotient, remainder = divmod(count, limit)
    pages = quotient + (remainder > 0)

    with tqdm.tqdm(total=count, desc='Retrieving') as progress_bar:
        for page in range(1, pages+1):
            response = requests.get(FILES_PREFIX + f'?limit={limit}&page={page}&fields[]=id&fields[]=title&sort[]=id&filter={{ "folder": {{ "_eq": "ce8c9263-e584-409b-bd99-ebf2453c6d38"}}}}', headers=headers)
            data = response.json()
            for item in data['data']:
                yield (item['id'], item['title'].replace('.tgz', ''))
                progress_bar.update(1)


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


def create_series_filter_item(series_filter_item: SeriesFilterItem, token: str, proxies: dict) -> tuple[str, str]:
    headers = get_headers(token)
    response = None
    try:
        if proxies is None:
            response = requests.post(SERIES_FILTER_PREFIX, headers=headers, json=[series_filter_item.dict()], timeout=15)
        else:
            response = requests.post(SERIES_FILTER_PREFIX, headers=headers, json=[series_filter_item.dict()], timeout=15, proxies=proxies)
        data = response.json()
        if 'data' in data:
            # flag = data['data'][0]['instance_hash']
            return f'{series_filter_item.instance_hash}'
        else:
            # flag = f'{series_filter_item.instance_name}: {str(data)}'
            return None
    except Exception as error:
        if response is not None:
            print(response)
            print(response.text)
        raise error


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


def insert_instance(parameter: tuple[pathlib.Path, pathlib.Path, str, bool, str, set[str], dict[str, str]]):
    sess = requests.session()
    sess.keep_alive = False
    (instance_dirpath, cache_dirpath, meta, with_attributes, since_version, paper, token, exist_instances, maps, proxy) = parameter
    if proxy is None:
        proxies = None
    else:
        proxies = {
            'http': proxy,
            'https': proxy
        }

    instance_filename = hash_string(instance_dirpath.name + f'-Paper:{paper}-With_Attributes:{with_attributes}', hash_algorithm='blake2b', digest_size=16)

    if instance_filename in exist_instances:
        return None

    headers = get_headers(token)

    response = None
    try:
        if proxies is None:
            response = requests.get(SERIES_FILTER_PREFIX+ f'?fields[]=instance_hash&filter[instance_hash][_eq]={instance_filename}', headers, timeout=15)
        else:
            response = requests.get(SERIES_FILTER_PREFIX+ f'?fields[]=instance_hash&filter[instance_hash][_eq]={instance_filename}', headers, timeout=15, proxies=proxies)
        data = response.json()
        assert len(data['data']) in {0, 1}
        if len(data['data']) == 1:
            return instance_filename
    except Exception as error:
        if response is not None:
            print(response)
            print(response.text)
        raise error

    meta_filepath = cache_dirpath.joinpath(instance_filename + '.json')
    instance_meta = generate_instance_meta(instance_dirpath, meta_filepath)

    series_filter_item = SeriesFilterItem(
        instance_hash=instance_filename,
        node_number=instance_meta['node_number'],
        edge_number=instance_meta['edge_number'],
        with_attributes=with_attributes,
        since_version=since_version,
        paper=paper,
        status='access',
        instance_meta=None,
        instance_tgz=maps[instance_filename]
    )

    instance_filename = create_series_filter_item(series_filter_item=series_filter_item, token=token, proxies=proxies)
    return instance_filename


def upload_instance(parameter: tuple[pathlib.Path, pathlib.Path, str, bool, str, dict[str, str]]):
    (instance_dirpath, cache_dirpath, meta, with_attributes, since_version, paper, token, exist_instances, proxy) = parameter
    sess = requests.session()
    sess.keep_alive = False
    if proxy is None:
        proxies = None
    else:
        proxies = {
            'http': proxy,
            'https': proxy
        }

    instance_filename = hash_string(instance_dirpath.name + f'-Paper:{paper}-With_Attributes:{with_attributes}', hash_algorithm='blake2b', digest_size=16)

    if instance_filename in exist_instances:
        return None, None

    headers = get_headers(token)

    response = None
    try:
        response = requests.get(FILES_PREFIX + f'?fields[]=id&filter[title][_eq]={instance_filename + ".tgz"}&filter[folder][_eq]=ce8c9263-e584-409b-bd99-ebf2453c6d38', headers)
        data = response.json()
        assert len(data['data']) in {0, 1}
        if len(data['data']) == 1:
            return data['data'][0]['id'], instance_filename
    except Exception as error:
        if response is not None:
            print(response)
            print(response.text)
        raise error

    archive_filepath = cache_dirpath.joinpath(instance_filename + '.tgz')
    tar_archive(instance_dirpath, archive_filepath, compress=True)

    with open(archive_filepath, 'rb') as archive_file:
        payload = dict()
        payload['storage'] = 'vultr'
        payload['folder'] = 'ce8c9263-e584-409b-bd99-ebf2453c6d38'
        payload['title'] = archive_filepath.name
        files = (
            ('file', (archive_filepath.name, archive_file, 'application/gzip')),
        )

        response = None
        try:
            if proxies is None:
                response = requests.post(FILES_PREFIX, headers=headers, data=payload, files=files, timeout=150)
            else:
                response = requests.post(FILES_PREFIX, headers=headers, data=payload, files=files, timeout=150, proxies=proxies)
            data = response.json()
        except Exception as error:
            if response is not None:
                print(response)
                print(response.text)
            raise error
        instance_tgz_id = data['data']['id']

    return instance_tgz_id, instance_filename


def main(dataset_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, memory_dirpath: pathlib.Path, worker_number: int = 4, meta: bool = False, with_attributes: bool = False, since_version: str = '0.0.0', paper: bool = False, token: str = None, proxy: str = None):
    logger.info(f'Checking Cache Directory Path: {cache_dirpath}')
    if cache_dirpath.is_dir():
        cache_content = [path for path in cache_dirpath.iterdir()]
        assert len(cache_content) == 0, 'You Need Specify An Empty Cache Directory.'
    else:
        create_dir(cache_dirpath)

    upload_fp = memory_dirpath.joinpath('upload')
    insert_fp = memory_dirpath.joinpath('insert')

    exist_instances = dict()
    if upload_fp.is_file():
        with open(upload_fp, 'r') as upload_file:
            for index, line in enumerate(upload_file):
                file_id, instance_filename = line.strip().split('--S--')
                exist_instances[instance_filename] = file_id
    logger.info(f'Already Uploaded {len(exist_instances)}.')

    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters: list[tuple[pathlib.Path, pathlib.Path]] = list()
    for path in sorted(list(dataset_dirpath.iterdir())):
        if path.is_dir():
            parameters.append((path, cache_dirpath, meta, with_attributes, since_version, paper, token, exist_instances, proxy))

    logger.info(f'Total Instances To Be Uploaded: {len(parameters)}')

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Uploading') as progress_bar:
            for index, (file_id, instance_filename) in enumerate(pool.imap_unordered(upload_instance, parameters), start=1):
                if file_id is not None and instance_filename is not None:
                    with open(upload_fp, 'a') as upload_file:
                        upload_file.write(f'{file_id}--S--{instance_filename}\n')
                progress_bar.update(1)
    delete_dir(cache_dirpath, only_clean=True)

    maps = dict()
    with open(upload_fp, 'r') as upload_file:
        for index, line in enumerate(upload_file):
            file_id, instance_filename = line.strip().split('--S--')
            maps[instance_filename] = file_id

    #====
    exist_instances = set()
    if insert_fp.is_file():
        with open(insert_fp, 'r') as insert_file:
            for index, line in enumerate(insert_file):
                exist_instances.add(line.strip())
    logger.info(f'Already Inserted {len(exist_instances)}.')

    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters: list[tuple[pathlib.Path, pathlib.Path]] = list()
    for path in sorted(list(dataset_dirpath.iterdir())):
        if path.is_dir():
            parameters.append((path, cache_dirpath, meta, with_attributes, since_version, paper, token, exist_instances, maps, proxy))

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Inserting') as progress_bar:
            for index, instance_filename in enumerate(pool.imap_unordered(insert_instance, parameters), start=1):
                if instance_filename is not None:
                    with open(insert_fp, 'a') as insert_file:
                        insert_file.write(f'{instance_filename}\n')
                progress_bar.update(1)