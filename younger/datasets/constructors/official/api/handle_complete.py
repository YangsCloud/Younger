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
requests.DEFAULT_RETRIES = 5


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


def create_series_complete_item(series_complete_item: SeriesCompleteItem, token: str, proxies: dict) -> tuple[bool, str]:
    headers = get_headers(token)
    response = None
    try:
        if proxies is None:
            response = requests.post(SERIES_COMPLETE_PREFIX, headers=headers, json=[series_complete_item.dict()], timeout=15)
        else:
            response = requests.post(SERIES_COMPLETE_PREFIX, headers=headers, json=[series_complete_item.dict()], timeout=15, proxies=proxies)
        data = response.json()
        if 'data' in data:
            # flag = data['data'][0]['instance_hash']
            return f'{series_complete_item.instance_name}'
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
    (instance_dirpath, cache_dirpath, since_version, paper, token, maps, proxy) = parameter
    if proxy is None:
        proxies = None
    else:
        proxies = {
            'http': proxy,
            'https': proxy
        }

    instance_filename = hash_string(instance_dirpath.name + f'-Paper:{paper}', hash_algorithm='blake2b', digest_size=16)

    # if instance_filename in exist_instances:
    #     return None

    headers = get_headers(token)

    response = None
    try:
        if proxies is None:
            response = requests.get(SERIES_COMPLETE_PREFIX+ f'?fields[]=instance_name&filter[instance_name][_eq]={instance_filename}', headers=headers, timeout=15)
        else:
            response = requests.get(SERIES_COMPLETE_PREFIX+ f'?fields[]=instance_name&filter[instance_name][_eq]={instance_filename}', headers=headers, timeout=15, proxies=proxies)
        data = response.json()
        assert len(data['data']) in {0, 1}
        if len(data['data']) == 1:
            return instance_filename, instance_dirpath.name
    except Exception as error:
        print('Error URL:', SERIES_COMPLETE_PREFIX+ f'?fields[]=instance_name&filter[instance_name][_eq]={instance_filename}')
        if response is not None:
            print(response)
            print(response.text)
        raise error

    meta_filepath = cache_dirpath.joinpath(instance_filename + '.json')
    instance_meta = generate_instance_meta(instance_dirpath, meta_filepath)

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
        instance_tgz=maps[instance_dirpath.name][0]
    )

    instance_filename = create_series_complete_item(series_complete_item=series_complete_item, token=token, proxies=proxies)
    return instance_filename, instance_dirpath.name


def upload_instance(parameter: tuple[pathlib.Path, pathlib.Path, str, bool, str, dict[str, str]]):
    sess = requests.session()
    sess.keep_alive = False
    (instance_dirpath, cache_dirpath, since_version, paper, token, proxy) = parameter
    if proxy is None:
        proxies = None
    else:
        proxies = {
            'http': proxy,
            'https': proxy
        }

    instance_filename = hash_string(instance_dirpath.name + f'-Paper:{paper}', hash_algorithm='blake2b', digest_size=16)

    # if instance_filename in exist_instances:
    #     return None, None

    headers = get_headers(token)

    response = None
    try:
        response = requests.get(FILES_PREFIX + f'?fields[]=id&filter[title][_eq]={instance_filename + ".tgz"}&filter[folder][_eq]=6ba43f7f-8cf1-49bb-894f-0ac75e3b5b0f', headers)
        data = response.json()
        assert len(data['data']) in {0, 1}
        if len(data['data']) == 1:
            return data['data'][0]['id'], instance_filename, instance_dirpath.name
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
        payload['folder'] = '6ba43f7f-8cf1-49bb-894f-0ac75e3b5b0f'
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
            instance_tgz_id = data['data']['id']
        except Exception as error:
            if response is not None:
                print(response)
                print(response.text)
            raise error

    return instance_tgz_id, instance_filename, instance_dirpath.name


def main(dataset_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, memory_dirpath: pathlib.Path, worker_number: int = 4, since_version: str = '0.0.0', paper: bool = False, token: str = None, proxy: str = None):
    logger.info(f'Checking Cache Directory Path: {cache_dirpath}')
    if cache_dirpath.is_dir():
        cache_content = [path for path in cache_dirpath.iterdir()]
        assert len(cache_content) == 0, 'You Need Specify An Empty Cache Directory.'
    else:
        create_dir(cache_dirpath)

    # logger.info(f'Retrieving Already Inserted Instances...')
    # exist_instances = set(read_series_complete_items(token))
    # logger.info(f'Retrieved Total {len(exist_instances)}.')

    if not memory_dirpath.is_dir():
        create_dir(memory_dirpath)

    upload_fp = memory_dirpath.joinpath('upload')
    insert_fp = memory_dirpath.joinpath('insert')

    exist_instances = dict()
    if upload_fp.is_file():
        with open(upload_fp, 'r') as upload_file:
            for index, line in enumerate(upload_file):
                file_id, instance_filename, instance_truename = line.strip().split('--S--')
                exist_instances[instance_truename] = (file_id, instance_filename)
    logger.info(f'Already Uploaded {len(exist_instances)}.')

    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters: list[tuple[pathlib.Path, pathlib.Path]] = list()
    for path in sorted(list(dataset_dirpath.iterdir())):
        if path.is_dir():
            if path.name not in exist_instances:
                parameters.append((path, cache_dirpath, since_version, paper, token, proxy))

    logger.info(f'Total Instances To Be Uploaded: {len(parameters)}')

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Uploading') as progress_bar:
            for index, (file_id, instance_filename, instance_truename) in enumerate(pool.imap_unordered(upload_instance, parameters), start=1):
                if file_id is not None and instance_filename is not None and instance_truename is not None:
                    with open(upload_fp, 'a') as upload_file:
                        upload_file.write(f'{file_id}--S--{instance_filename}--S--{instance_truename}\n')
                progress_bar.update(1)
    delete_dir(cache_dirpath, only_clean=True)

    maps = dict()
    with open(upload_fp, 'r') as upload_file:
        for index, line in enumerate(upload_file):
            file_id, instance_filename, instance_truename = line.strip().split('--S--')
            maps[instance_truename] = (file_id, instance_filename)

    #====
    exist_instances = dict()
    if insert_fp.is_file():
        with open(insert_fp, 'r') as insert_file:
            for index, line in enumerate(insert_file):
                instance_filename, instance_truename = line.strip().split('--S--')
                exist_instances[instance_truename] = instance_filename
    logger.info(f'Already Inserted {len(exist_instances)}.')

    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters: list[tuple[pathlib.Path, pathlib.Path]] = list()
    for path in sorted(list(dataset_dirpath.iterdir())):
        if path.is_dir():
            if path.name not in exist_instances:
                parameters.append((path, cache_dirpath, since_version, paper, token, maps, proxy))

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Inserting') as progress_bar:
            for index, (instance_filename, instance_truename) in enumerate(pool.imap_unordered(insert_instance, parameters), start=1):
                if instance_filename is not None and instance_truename is not None:
                    with open(insert_fp, 'a') as insert_file:
                        insert_file.write(f'{instance_filename}--S--{instance_truename}\n')
                progress_bar.update(1)
