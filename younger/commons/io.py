#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-06 20:34
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import pickle
import psutil
import shutil
import tarfile
import pathlib

from younger.commons.hash import hash_bytes
from younger.commons.logging import logger


def create_dir(dirpath: pathlib.Path) -> None:
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
    except Exception as exception:
        logger.error(f'An Error occurred while creating the directory: {str(exception)}')
        raise exception

    return


def delete_dir(dirpath: pathlib.Path, only_clean: bool = False):
    for filepath in dirpath.iterdir():
        if filepath.is_dir():
            shutil.rmtree(filepath)
        if filepath.is_file():
            os.remove(filepath)

    if not only_clean:
        os.rmdir(dirpath)


def tar_archive(ri: pathlib.Path | list[pathlib.Path], archive_filepath: pathlib.Path, compress: bool = True):
    # ri - read in
    if compress:
        mode = 'w:gz'
    else:
        mode = 'w'

    with tarfile.open(archive_filepath, mode=mode, dereference=False) as tar:
        if isinstance(ri, list):
            for path in ri:
                tar.add(path, arcname=os.path.basename(path))
        if isinstance(ri, pathlib.Path):
            tar.add(ri, arcname=os.path.basename(ri))


def tar_extract(archive_filepath: pathlib.Path, wo: pathlib.Path, compress: bool = True):
    # wo - write out
    if compress:
        mode = 'r:gz'
    else:
        mode = 'r'

    with tarfile.open(archive_filepath, mode=mode, dereference=False) as tar:
        tar.extractall(wo)


def load_json(filepath: pathlib.Path) -> object:
    try:
        with open(filepath, 'r') as file:
            serializable_object = json.load(file)
    except Exception as exception:
        logger.error(f'An Error occurred while reading serializable object from the \'json\' file: {str(exception)}')
        raise exception

    return serializable_object


def save_json(serializable_object: object, filepath: pathlib.Path, indent: int | str | None = None) -> None:
    try:
        create_dir(filepath.parent)
        with open(filepath, 'w') as file:
            json.dump(serializable_object, file, indent=indent)
    except Exception as exception:
        logger.error(f'An Error occurred while writing serializable object into the \'json\' file: {str(exception)}')
        raise exception

    return


def load_pickle(filepath: pathlib.Path) -> object:
    try:
        with open(filepath, 'rb') as file:
            safety_data = pickle.load(file)

        assert hash_bytes(safety_data['main']) == safety_data['checksum']
        serializable_object = pickle.loads(safety_data['main'])
    except Exception as exception:
        logger.error(f'An Error occurred while reading serializable object from the \'pickle\' file: {str(exception)}')
        raise exception

    return serializable_object


def save_pickle(serializable_object: object, filepath: pathlib.Path) -> None:
    try:
        create_dir(filepath.parent)
        serialized_object = pickle.dumps(serializable_object)
        safety_data = dict(
            main=serialized_object,
            checksum=hash_bytes(serialized_object)
        )
        with open(filepath, 'wb') as file:
            pickle.dump(safety_data, file)
    except Exception as exception:
        logger.error(f'An Error occurred while writing serializable object into the \'pickle\' file: {str(exception)}')
        raise exception

    return


def get_disk_free_size(path: pathlib.Path) -> int:
    disk_usage = psutil.disk_usage(path)
    return disk_usage.free


def get_path_size(path: pathlib.Path) -> int:
    if path.is_file():
        return get_file_size(path)
    else:
        return get_dir_size(path)


def get_file_size(filepath: pathlib.Path) -> int:
    return os.path.getsize(filepath)


def get_dir_size(dirpath: pathlib.Path) -> int:
    total_size = 0
    for root, _, files in os.walk(dirpath):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    return total_size