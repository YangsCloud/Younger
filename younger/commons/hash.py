#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 16:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import hashlib


def hash_file(filepath: pathlib.Path | str, block_size: int = 8192, hash_algorithm: str = "SHA256", digest_size: int | None = None) -> str:
    filepath = pathlib.Path(filepath) if isinstance(filepath, str) else filepath
    hasher = hashlib.new(hash_algorithm) if digest_size is None else hashlib.new(hash_algorithm, digest_size=digest_size)
    with open(filepath, 'rb') as file:
        while True:
            block = file.read(block_size)
            if len(block) == 0:
                break
            hasher.update(block)

    return str(hasher.hexdigest())


def hash_bytes(byte_string: bytes, hash_algorithm: str = "SHA256", digest_size: int | None = None) -> str:
    hasher = hashlib.new(hash_algorithm) if digest_size is None else hashlib.new(hash_algorithm, digest_size=digest_size)
    hasher.update(byte_string)

    return str(hasher.hexdigest())


def hash_string(string: str, hash_algorithm: str = "SHA256", digest_size: int | None = None) -> str:
    hasher = hashlib.new(hash_algorithm) if digest_size is None else hashlib.new(hash_algorithm, digest_size=digest_size)
    hasher.update(string.encode('utf-8'))

    return str(hasher.hexdigest())


def hash_strings(strings: list[str], hash_algorithm: str = "SHA256", digest_size: int | None = None) -> str:
    hasher = hashlib.new(hash_algorithm) if digest_size is None else hashlib.new(hash_algorithm, digest_size=digest_size)
    for string in strings:
        hasher.update(string.encode('utf-8'))

    return str(hasher.hexdigest())