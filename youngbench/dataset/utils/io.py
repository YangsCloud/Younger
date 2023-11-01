#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-14 14:55
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import onnx
import json
import hashlib
import pathlib

from onnx import version_converter
from typing import List, Union

from youngbench.logging import logger
from youngbench.constants import ONNX


def hash_binary(filepath: pathlib.Path, block_size: int = 8192, hash_algorithm: str = "SHA256") -> str:
    hasher = hashlib.new(hash_algorithm)
    with open(filepath, 'rb') as file:
        while True:
            block = file.read(block_size)
            if len(block) == 0:
                break
            hasher.update(block)

    return str(hasher.hexdigest())


def hash_bytes(byte_string: bytes, hash_algorithm: str = "SHA256") -> str:
    hasher = hashlib.new(hash_algorithm)
    hasher.update(byte_string)

    return str(hasher.hexdigest())


def hash_strings(strings: List[str], hash_algorithm: str = "SHA256") -> str:
    hasher = hashlib.new(hash_algorithm)
    for string in strings:
        hasher.update(string.encode('utf-8'))

    return str(hasher.hexdigest())


def hash_string(string: str, hash_algorithm: str = "SHA256") -> str:
    hasher = hashlib.new(hash_algorithm)
    hasher.update(string.encode('utf-8'))

    return str(hasher.hexdigest())


def create_dir(dirpath: pathlib.Path) -> None:
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f'An Error occurred while creating the directory: {str(e)}')
        sys.exit(1)

    return


def read_json(filepath: pathlib.Path) -> object:
    try:
        with open(filepath, 'r') as file:
            serializable_object = json.load(file)
    except Exception as e:
        logger.error(f'An Error occurred while reading serializable object from the file: {str(e)}')
        sys.exit(1)

    return serializable_object


def write_json(serializable_object: object, filepath: pathlib.Path) -> None:
    try:
        create_dir(filepath.parent)
        with open(filepath, 'w') as file:
            json.dump(serializable_object, file)
    except Exception as e:
        logger.error(f'An Error occurred while writing serializable object into the file: {str(e)}')
        sys.exit(1)

    return


def check_onnx_model(model_handler: Union[onnx.ModelProto, pathlib.Path]) -> bool:
    if isinstance(model_handler, pathlib.Path):
        model_handler = str(model_handler)
    try:
        onnx.checker.check_model(model_handler)
        check_result = True
    except onnx.checker.ValidationError as check_error:
        logger.warn(f'The ONNX Model is invalid: {check_error}')
        check_result = False
    except Exception as error:
        logger.error(f'An error occurred while checking the ONNX model: {error}')
        sys.exit(1)
    return check_result


def load_onnx_model(model_filepath: pathlib.Path) -> onnx.ModelProto:
    model = onnx.load(model_filepath)
    return model


def save_onnx_model(onnx_model: onnx.ModelProto, model_filepath: pathlib.Path) -> None:
    create_dir(model_filepath.parent)
    onnx.save(onnx_model, model_filepath)
    return


def clean_onnx_model(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    onnx_model = version_converter.convert_version(onnx_model, ONNX.OPSetVersions[-1])
    return onnx_model