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
import shutil
import hashlib
import pathlib
import networkx

from typing import List, Iterable

from youngbench.dataset.logging import logger


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
        dirpath.mkdir()
        logger.info(f'Directory \"{dirpath}\" created successfully')
    except FileExistsError:
        logger.warn(f'The directory \"{dirpath}\" exists.')
    except Exception as e:
        logger.error(f'An Error occurred while creating the directory: {str(e)}')
        sys.exit(1)

    return


def backup_file(filepath: pathlib.Path) -> None:
    try:
        shutil.copy(filepath, filepath+'.backup')
        logger.info(f'The backup file \"{filepath}.backup\" created successfully.')
    except Exception as e:
        logger.error(f'An Error occurred while creating the backup file: {str(e)}')
        sys.exit(1)

    return


def read_json(filepath: pathlib.Path) -> object:
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            serializable_object = json.load(file)
        logger.info(f'The serializable object reading from the file \"{filepath}\" successfully.')
    except Exception as e:
        logger.error(f'An Error occurred while reading serializable object from the file: {str(e)}')
        sys.exit(1)

    return serializable_object


def write_json(serializable_object: object, filepath: pathlib.Path) -> None:
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(serializable_object, file)
        logger.info(f'The serializable object writing into the file \"{filepath}\" successfully.')
    except Exception as e:
        logger.error(f'An Error occurred while writing serializable object into the file: {str(e)}')
        sys.exit(1)

    return


def load_onnx_model(model_filepath: pathlib.Path) -> onnx.ModelProto:
    try:
        onnx.checker.check_model(str(model_filepath))
        model = onnx.load(model_filepath)
    except onnx.checker.ValidationError as e:
        logger.error(f'The onnx model is invalid: {e}')
        sys.exit(2)
    except Exception as e:
        logger.error(f'An Error occurred while loading onnx model: {str(e)}')
        sys.exit(1)

    return model


def save_onnx_model(model: onnx.ModelProto, model_filepath: pathlib.Path) -> None:
    onnx.save(model, model_filepath)

    return


def get_opset_version(onnx_model: onnx.ModelProto):
    for opset_info in onnx_model.opset_import:
        if opset_info.domain == "":
            opset_version = opset_info.version
            break

    return opset_version


def isomorphic(prototype_lhs: networkx.DiGraph, prototype_rhs: networkx.DiGraph) -> bool:
    isomorphic = True
    isomorphic = isomorphic and networkx.vf2pp_is_isomorphic(prototype_lhs, prototype_rhs, node_label='op_type')
    isomorphic = isomorphic and networkx.vf2pp_is_isomorphic(prototype_lhs, prototype_rhs, node_label='op_domain')
    return isomorphic