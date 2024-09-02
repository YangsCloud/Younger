#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-08-30 14:49
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib

from xml.etree import ElementTree

from younger.commons.io import tar_extract
from younger.commons.logging import logger
from younger.commons.download import download


def prepare_phoronix_onnx(download_xml: pathlib.Path, download_dir: pathlib.Path):
    xml_tree = ElementTree.parse(download_xml)

    xml_root = xml_tree.getroot()

    xml_downloads = xml_root.find('Downloads')

    workloads = list()

    for package in xml_downloads:
        workload_name = package.find('FileName').text
        workload_size = package.find('FileSize').text
        workload_link = package.find('URL').text
        workloads.append(
            dict(
                name = workload_name,
                size = workload_size,
                link = workload_link,
            )
        )


    tar_filepaths = list()
    os.makedirs(download_dir, exist_ok=True)
    for index, workload in enumerate(workloads, start=1):
        logger.info(f' v {index}. Now download {workload["name"]} (Size: {int(workload["size"])//(1024*1024)}MB)...')
        workload_link = workload["link"].replace('blob', 'raw')
        tar_filepath = download(workload_link, download_dir, force=False)
        tar_filepaths.append(tar_filepath)
        logger.info(' ^ Done')

    logger.info(' = Uncompress All Tars')
    for index, tar_filepath in enumerate(tar_filepaths, start=1):
        logger.info(f' v {index}. Uncompressing {tar_filepath}...')
        tar_extract(tar_filepath, download_dir.joinpath('models'))
        logger.info(' ^ Done')
