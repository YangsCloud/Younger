#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-14 14:55
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import onnx
import json
import pickle
import shutil
import pathlib
import tarfile

from onnx import version_converter

from younger.commons.logging import logger

from younger.datasets.utils.hash import hash_bytes
from younger.datasets.utils.constants import ONNX


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


def check_model(model_handler: onnx.ModelProto | pathlib.Path) -> bool:
    assert isinstance(model_handler, onnx.ModelProto) or isinstance(model_handler, pathlib.Path)
    # Change Due To Hash May Lead OOM.
    def check_with_internal() -> str or None:
        model = model_handler
        if len(model.graph.node) == 0:
            check_result = False
        else:
            onnx.checker.check_model(model)
            #check_result = hash_bytes(model)
            check_result = True
        return check_result

    def check_with_external() -> str or None:
        onnx.checker.check_model(str(model_handler))
        #model = onnx.load(str(model_handler))
        #check_result = hash_bytes(model.SerializeToString())
        check_result = True

        return check_result

    try:
        if isinstance(model_handler, onnx.ModelProto):
            return check_with_internal()
        if isinstance(model_handler, pathlib.Path):
            return check_with_external()
    except onnx.checker.ValidationError as check_error:
        logger.warn(f'The ONNX Model is invalid: {check_error}')
        check_result = False
    except Exception as error:
        logger.error(f'An error occurred while checking the ONNX model: {error}')
        sys.exit(1)
    return check_result


def load_model(model_filepath: pathlib.Path) -> onnx.ModelProto:
    model = onnx.load(model_filepath, load_external_data=False)
    return model


def save_model(model: onnx.ModelProto, model_filepath: pathlib.Path) -> None:
    create_dir(model_filepath.parent)
    onnx.save(model, model_filepath)
    return


def clean_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = version_converter.convert_version(model, ONNX.OPSetVersions[-1])
    return model