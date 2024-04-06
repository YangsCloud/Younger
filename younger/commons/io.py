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
import shutil
import tarfile
import pathlib

from younger.commons.hash import hash_bytes
from younger.commons.logging import logger


def create_dir(dirpath: pathlib.Path) -> None:
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f'An Error occurred while creating the directory: {str(e)}')
        sys.exit(1)

    return


def delete_dir(dirpath: pathlib.Path):
    for filepath in dirpath.iterdir():
        if filepath.is_dir():
            shutil.rmtree(filepath)
        if filepath.is_file():
            os.remove(filepath)

    os.rmdir(dirpath)


def tar_archive(dirpath: pathlib.Path, archive_filepath: pathlib.Path, compress: bool = True):
    if compress:
        mode = 'w:gz'
    else:
        mode = 'w'

    with tarfile.open(archive_filepath, mode=mode, dereference=False) as tar:
        tar.add(dirpath, arcname=os.path.basename(dirpath))

def tar_extract(archive_filepath: pathlib.Path, dirpath: pathlib.Path, compress: bool = True):
    if compress:
        mode = 'r:gz'
    else:
        mode = 'r'

    with tarfile.open(archive_filepath, mode=mode, dereference=False) as tar:
        tar.extractall(dirpath)


def load_json(filepath: pathlib.Path) -> object:
    try:
        with open(filepath, 'r') as file:
            serializable_object = json.load(file)
    except Exception as e:
        logger.error(f'An Error occurred while reading serializable object from the \'json\' file: {str(e)}')
        sys.exit(1)

    return serializable_object


def save_json(serializable_object: object, filepath: pathlib.Path) -> None:
    try:
        create_dir(filepath.parent)
        with open(filepath, 'w') as file:
            json.dump(serializable_object, file)
    except Exception as e:
        logger.error(f'An Error occurred while writing serializable object into the \'json\' file: {str(e)}')
        sys.exit(1)

    return


def load_pickle(filepath: pathlib.Path) -> object:
    try:
        with open(filepath, 'rb') as file:
            safety_data = pickle.load(file)
        
        assert hash_bytes(safety_data['main']) == safety_data['checksum']
        serializable_object = pickle.loads(safety_data['main'])
    except Exception as e:
        logger.error(f'An Error occurred while reading serializable object from the \'pickle\' file: {str(e)}')
        sys.exit(1)

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
    except Exception as e:
        logger.error(f'An Error occurred while writing serializable object into the \'pickle\' file: {str(e)}')
        sys.exit(1)

    return

