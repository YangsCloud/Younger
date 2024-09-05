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


import pathlib

from xml.etree import ElementTree

from younger.commons.io import tar_extract
from younger.commons.logging import logger
from younger.commons.download import download

from younger.datasets.modules import Instance


SUPPORT_VERSION = {
    'phoronix'
}


def phoronix_prepare(bench_dirpath: pathlib.Path, dataset_dirpath: pathlib.Path) -> list[Instance]:
    xml_tree = ElementTree.parse(bench_dirpath.joinpath("downloads.xml"))

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
    for index, workload in enumerate(workloads, start=1):
        logger.info(f' v {index}. Now download {workload["name"]} (Size: {int(workload["size"])//(1024*1024)}MB)...')
        workload_link = workload["link"].replace('blob', 'raw')
        tar_filepath = download(workload_link, bench_dirpath, force=False)
        tar_filepaths.append(tar_filepath)
        logger.info(' ^ Done')

    logger.info(' = Uncompress All Tars')
    for index, tar_filepath in enumerate(tar_filepaths, start=1):
        logger.info(f' v {index}. Uncompressing {tar_filepath}...')
        tar_extract(tar_filepath, bench_dirpath)
        logger.info(' ^ Done')

    logger.info(' = Extracting All Instances')
    instances: list[Instance] = list()
    for model_dirpath in bench_dirpath.iterdir():
        if not model_dirpath.is_dir():
            continue
        model_name = model_dirpath.name
        logger.info(f' v Now processing model: {model_name}')
        model_filepaths = [model_filepath for model_filepath in model_dirpath.rglob('*.onnx')]
        assert len(model_filepaths) == 1

        model_filepath = model_filepaths[0]
        instance_dirpath = dataset_dirpath.joinpath(model_name)
        if instance_dirpath.is_dir():
            logger.info(f'   Instance Alreadly Exists: {instance_dirpath}')
            instance = Instance()
            instance.load(instance_dirpath)
            instances.append(instance)
        else:
            logger.info(f'   Model Filepath: {model_filepath}')
            logger.info(f'   Extracting Instance ...')
            instance = Instance(model_filepath)
            instances.append(instance)
            logger.info(f'   Extracted')
            instance.save(instance_dirpath)
            logger.info(f'   Instance saved into {instance_dirpath}')
        logger.info(f' ^ Done')
    return instances


def main(bench_dirpath: pathlib.Path, dataset_dirpath: pathlib.Path, version: str):
    assert version in SUPPORT_VERSION
    prepared = False
    if version == 'phoronix':
        instances = phoronix_prepare(bench_dirpath, dataset_dirpath)
        prepared = True

    if prepared:
        logger.info(f' = Extracted Dataset From Benchmark. Dataset Size: {len(instances)}')