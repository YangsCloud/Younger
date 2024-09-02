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

    print(f'Downloading {url}')

    create_dir(dirpath)

    block_size = 1024
    if filepath.is_file():
        resume_byte_pos = filepath.stat().st_size
    else:
        resume_byte_pos = 0

    response = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(response.headers.get('Content-Length', '0'))

    headers = {'Content-Length': '0', 'Range': f'bytes={resume_byte_pos}-'}
    response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
    if not force and (resume_byte_pos == total_size or total_size == 0):
        print(f'File is already downloaded: {filename}')
        return filepath

    if resume_byte_pos < total_size:
        with tqdm.tqdm(total=total_size, initial=resume_byte_pos, unit="iB", unit_scale=True, unit_divisor=1024, desc=filename) as progress_bar:
            with fsspec.open(filepath, "ab") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))

    return filepath