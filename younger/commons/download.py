#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-08-30 16:26
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import ssl
import tqdm
import fsspec
import pathlib
import requests

import urllib.request

from younger.commons.io import create_dir


def download(url: str, dirpath: pathlib.Path, force: bool = True):
    r"""Downloads the content of an URL to a specific directory path.

    Args:
        url (str): The URL.
        dirpath (pathlib.Path): The folder.
    """
    filename = url.rpartition('/')[2]
    filename = filename if filename[0] == '?' else filename.split('?')[0]

    filepath = dirpath.joinpath(filename)

    if filepath.is_file() and not force:  # pragma: no cover
        print(f'File already exists: {filename}')
        return filepath

    print(f'Downloading {url}')

    create_dir(dirpath)

    response = requests.get(url, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024, desc=filename) as progress_bar:
        with fsspec.open(filepath, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))

    return filepath