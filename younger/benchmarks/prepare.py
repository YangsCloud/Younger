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
import networkx

from typing import Literal
from xml.etree import ElementTree
from optimum.exporters.onnx import main_export

from younger.commons.io import tar_extract, create_dir, save_json
from younger.commons.logging import logger
from younger.commons.download import download

from younger.datasets.modules import Dataset, Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node, get_onnx_opset_version


SUPPORT_VERSION = {
    'younger',
    'convert',
    'phoronix',
    'mlperf_v4.1',
    'mlperf_v0.5'
}


def purify_instance(instance: Instance, model_source: str, model_name: str) -> Instance:
    standardized_graph = Network.standardize(instance.network.graph)
    for node_index in standardized_graph.nodes():
        operator = standardized_graph.nodes[node_index]['features']['operator']
        attributes = standardized_graph.nodes[node_index]['features']['attributes']
        standardized_graph.nodes[node_index]['features']['attributes'] = get_complete_attributes_of_node(attributes, operator['op_type'], operator['domain'], get_onnx_opset_version())

    standardized_graph.graph.clear()

    instance.setup_network(Network(standardized_graph))

    cleansed_graph = networkx.DiGraph()
    cleansed_graph.add_nodes_from(instance.network.graph.nodes(data=True))
    cleansed_graph.add_edges_from(instance.network.graph.edges(data=True))
    for node_index in cleansed_graph.nodes():
        cleansed_graph.nodes[node_index]['operator'] = cleansed_graph.nodes[node_index]['features']['operator']
    graph_hash = Network.hash(cleansed_graph, node_attr='operator')

    instance.clean_labels()
    instance.setup_labels(
        dict(
            model_sources = [model_source],
            model_name = [model_name],
            hash = graph_hash,
            downloads=[], likes=[], tags=[], evaluations=[]
        )
    )

    return instance


def hf_hub_download(r_path: str, l_path: str, ignore_patterns: list[str] | None = None):
    from huggingface_hub import HfFileSystem
    hf_file_system = HfFileSystem()

    def ignore(string: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            if pattern in string:
                return True
        return False

    if hf_file_system.isdir(r_path):
        logger.info(f'     -> A dirpath is provided, now downloading the dir ...')
        for r_child_path in hf_file_system.glob(os.path.join(r_path, '**'), detail=False):
            if hf_file_system.isfile(r_child_path):
                relative_path = os.path.relpath(r_child_path, start=r_path)
                l_child_path = os.path.join(l_path, relative_path)
                l_child_parent = os.path.dirname(l_child_path)
                child_link = hf_file_system.url(r_child_path)
                if ignore_patterns is not None and ignore(child_link, ignore_patterns):
                    logger.info(f'Skip. Ignore Pattern Match.')
                else:
                    create_dir(l_child_parent)
                    download(child_link, pathlib.Path(l_child_parent), force=False)
        logger.info(f'     -> Done')

    if hf_file_system.isfile(r_path):
        logger.info(f'     -> A filepath is provided, now downloading the file ...')
        link = hf_file_system.url(r_path)
        if ignore(link, ignore_patterns):
            logger.info(f'Skip. Ignore Pattern Match.')
        else:
            download(link, pathlib.Path(l_path), force=False)
        logger.info(f'     -> Done')


def mlperf_prepare(bench_dirpath: pathlib.Path, dataset_dirpath: pathlib.Path, release: str, direct: Literal['instance', 'onnx', 'both'] | None = None) -> list[Instance]:
    if direct:
        if release == 'v4.1':
            r_dirpath = 'datasets/AIJasonYoung/YoungBench-Assets/competitors/mlperf_v4.1'
        if release == 'v0.5':
            r_dirpath = 'datasets/AIJasonYoung/YoungBench-Assets/competitors/mlperf_v0.5'

        r_instances_dirpath = os.path.join(r_dirpath, 'instances')
        r_onnxs_dirpath = os.path.join(r_dirpath, 'onnxs')

        instances: list[Instance] = list()
        if direct in {'instance', 'both'}:
            logger.info(f' = Getting Instances From Official Repository ...')
            hf_hub_download(r_instances_dirpath, str(dataset_dirpath.absolute()), ['gpt-j-fp32', 'stable-diffusion-xl-fp32'])
            for path in dataset_dirpath.iterdir():
                instance = Instance()
                instance.load(path)
                instances.append(instance)

        if direct in {'onnx', 'both'}:
            logger.info(f' = Getting ONNXs ...')
            hf_hub_download(r_onnxs_dirpath, str(bench_dirpath.absolute()), ['gpt-j-fp32', 'stable-diffusion-xl-fp32'])

        if len(instances) == 0:
            pass
        else:
            logger.info(' = Not Retrieved Any Instances.')
    else:
        if release == 'v4.1':
            onnx_links = {
                'resnet50-v1.5-fp32': 'https://zenodo.org/record/4735647/files/resnet50_v1.onnx',
                'retinanet-800x800-fp32': 'https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx',
                'BERT-Large-int8': 'https://zenodo.org/records/3750364/files/bert_large_v1_1_fake_quant.onnx',
                'BERT-Large-fp32': 'https://zenodo.org/records/3733910/files/model.onnx',
                '3d-unet-fp32': 'https://zenodo.org/records/5597155/files/3dunet_kits19_128x128x128.onnx',
                # 'gpt-j-fp32': {'model_id': 'EleutherAI/gpt-j-6b', 'task': 'auto'},
                # 'stable-diffusion-xl-fp32': {'model_id': 'stabilityai/stable-diffusion-xl-base-1.0', 'task': 'stable-diffusion-xl'},
            }
        if release == 'v0.5':
            onnx_links = {
                'resnet50-v1.5-fp32': 'https://zenodo.org/record/2592612/files/resnet50_v1.onnx',
                'mobilenet-v1-fp32': 'https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx',
                # 'mobilenet-v1-int8': 'https://zenodo.org/record/3353417/files/Quantized%20MobileNet.zip', # Wrong Resources. Not An ONNX Model.
                'ssd-mobilenet-300x300-fp32': 'https://zenodo.org/record/3163026/files/ssd_mobilenet_v1_coco_2018_01_28.onnx',
                'ssd-resnet34-1200x1200-fp32': 'https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx',
            }

        for index, (name, link) in enumerate(onnx_links.items()):
            logger.info(f' v {index}. Now getting {name} ...')
            if isinstance(link, dict):
                logger.info(f'   - Getting & Converting from HuggingFace ...')
                model_dirpath = bench_dirpath.joinpath(f'{name}')
                if model_dirpath.is_dir():
                    logger.info(f'   - Already Getted.')
                else:
                    main_export(link['model_id'], model_dirpath, task=link['task'], monolith=True, trust_remote_code=True)
            else:
                logger.info(f'   - Getting from MLCommons ...')
                download(link, bench_dirpath, filename=f'{name}.onnx', force=False)
            logger.info(f' ^ Done')

        logger.info(' = Extracting All Instances')
        instances: list[Instance] = list()
        for index, (name, link) in enumerate(onnx_links.items()):
            logger.info(f' v Now processing model: {name}')
            if isinstance(link, dict):
                model_dirpath = bench_dirpath.joinpath(f'{name}')
                model_filepaths = [model_filepath for model_filepath in model_dirpath.rglob('*.onnx')]
            else:
                model_filepaths = [bench_dirpath.joinpath(f'{name}.onnx')]
            logger.info(f'   This ONNX model contains total {len(model_filepaths)} sub models')
            for index, model_filepath in enumerate(model_filepaths):
                model_name = f'{name}-{index}'
                instance_dirpath = dataset_dirpath.joinpath(model_name)
                if instance_dirpath.is_dir():
                    logger.info(f'    - Instance Alreadly Exists: {instance_dirpath}')
                    instance = Instance()
                    instance.load(instance_dirpath)
                    instances.append(instance)
                else:
                    logger.info(f'      Model Filepath: {model_filepath}')
                    logger.info(f'      Extracting Instance ...')
                    instance = Instance(model_filepath)
                    instance = purify_instance(instance, f'MLPerf_{release}', f'{model_name}')
                    instances.append(instance)
                    logger.info(f'    - Extracted')
                    instance.save(instance_dirpath)
                    logger.info(f'    - No.{index}. Instance saved into {instance_dirpath}')
            logger.info(f' ^ Done')
    return instances


def phoronix_prepare(bench_dirpath: pathlib.Path, dataset_dirpath: pathlib.Path, direct: Literal['instance', 'onnx', 'both'] | None = None) -> list[Instance]:
    if direct:
        r_dirpath = 'datasets/AIJasonYoung/YoungBench-Assets/competitors/phoronix'
        r_instances_dirpath = os.path.join(r_dirpath, 'instances')
        r_onnxs_dirpath = os.path.join(r_dirpath, 'onnxs')

        instances: list[Instance] = list()
        if direct in {'instance', 'both'}:
            logger.info(f' = Getting Instances From Official Repository ...')
            hf_hub_download(r_instances_dirpath, str(dataset_dirpath.absolute()))
            for path in dataset_dirpath.iterdir():
                instance = Instance()
                instance.load(path)
                instances.append(instance)

        if direct in {'onnx', 'both'}:
            logger.info(f' = Getting ONNXs ...')
            hf_hub_download(r_onnxs_dirpath, str(bench_dirpath.absolute()))

        if len(instances) == 0:
            pass
        else:
            logger.info(' = Not Retrieved Any Instances.')

    else:
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
            logger.info(f' v {index}. Now downloading {workload["name"]} (Size: {int(workload["size"])//(1024*1024)}MB)...')
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

            model_filepath = f'{model_filepaths[0]}-0'
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
                instance = purify_instance(instance, f'Phoronix', f'{model_name}')
                instances.append(instance)
                logger.info(f'   Extracted')
                instance.save(instance_dirpath)
                logger.info(f'   Instance saved into {instance_dirpath}')
            logger.info(f' ^ Done')
    return instances


def convert_prepare(bench_dirpath: pathlib.Path, dataset_dirpath: pathlib.Path) -> list[Instance]:
    logger.info(' = Extracting All Instances')
    instances: list[Instance] = list()
    for index, model_path in enumerate(bench_dirpath.iterdir()):
        model_name = model_path.stem
        logger.info(f' v Now processing model: {model_name}')
        if model_path.is_dir():
            model_dirpath = bench_dirpath.joinpath(f'{model_name}')
            model_filepaths = [model_filepath for model_filepath in model_dirpath.rglob('*.onnx')]
        else:
            if model_path.name.endswith('.onnx'):
                model_filepaths = [bench_dirpath.joinpath(f'{model_name}.onnx')]
            else:
                continue
        logger.info(f'   This ONNX model contains total {len(model_filepaths)} sub models')
        for index, model_filepath in enumerate(model_filepaths):
            model_name = f'{model_name}-{index}'
            instance_dirpath = dataset_dirpath.joinpath(model_name)
            if instance_dirpath.is_dir():
                logger.info(f'    - Instance Alreadly Exists: {instance_dirpath}')
                instance = Instance()
                instance.load(instance_dirpath)
                instances.append(instance)
            else:
                logger.info(f'      Model Filepath: {model_filepath}')
                logger.info(f'      Extracting Instance ...')
                instance = Instance(model_filepath)
                instance = purify_instance(instance, f'Convert', f'{model_name}')
                instances.append(instance)
                logger.info(f'    - Extracted')
                instance.save(instance_dirpath)
                logger.info(f'    - No.{index}. Instance saved into {instance_dirpath}')
        logger.info(f' ^ Done')
    return instances


def younger_prepare(bench_dirpath: pathlib.Path, dataset_dirpath: pathlib.Path) -> list[Instance]:
    YOUNGER_V_PAPER_FILTER_DETAIL_URL="https://datasets.yangs.cloud/public/assets/622387f1-dee7-4019-b538-34f341eb2914?download="
    YOUNGER_V_PAPER_FILTER_DETAIL_NAME = 'detailed_filter_series_without_attributes_paper'

    with open(bench_dirpath.joinpath('version')) as version_file:
        version = version_file.read().strip()

    if version == 'paper':
        younger_link = YOUNGER_V_PAPER_FILTER_DETAIL_URL
        younger_name = YOUNGER_V_PAPER_FILTER_DETAIL_NAME
    else:
        assert version in {'paper'}, f'Must Specify A Valid Version of Younger. Now Only Support: ["paper"] .'

    logger.info(f' v Downloading Younger ...')
    tar_filepath = download(younger_link, bench_dirpath, filename=f'{younger_name}.tgz', force=False)
    logger.info(f' ^ Done')

    younger_dirpath = dataset_dirpath.joinpath(younger_name)
    logger.info(f' v Uncompressing ...')
    if younger_dirpath.is_dir():
        logger.info(f'   Skip. Already Uncompressed.')
    else:
        tar_extract(tar_filepath, dataset_dirpath)
    logger.info(f' ^ Done')

    logger.info(f' v Checking all Instances in Younger ...')
    instances = [instance for instance in Dataset.load_instances(younger_dirpath)]
    logger.info(f' ^ Done')

    logger.info(f' v Saving Hash2Name (Instance Hash -> Model Names) ...')
    hash2names = dict()
    for instance in instances:
        hash2names[instance.labels['hash']] = instance.labels['model_name']
    hash2name_filepath = dataset_dirpath.joinpath('hash2names.json')
    save_json(hash2names, hash2name_filepath, indent=2)
    logger.info(f' ^ Done')
    return instances


def main(bench_dirpath: pathlib.Path, dataset_dirpath: pathlib.Path, version: str, direct: Literal['instance', 'onnx', 'both'] | None = None):
    assert version in SUPPORT_VERSION
    prepared = False
    if version == 'younger':
        instances = younger_prepare(bench_dirpath, dataset_dirpath)
        prepared = True

    if version == 'convert':
        # Convert Arbitrary Benchmarks (Any Dir Contains ONNX Modles)
        instances = convert_prepare(bench_dirpath, dataset_dirpath)
        prepared = True

    if version == 'phoronix':
        # One Get Phoronix ONNXs or Instances or Both
        instances = phoronix_prepare(bench_dirpath, dataset_dirpath, direct)
        prepared = True

    if version == 'mlperf_v4.1':
        # One Get MLPerfv4.1 ONNXs or Instances or Both
        instances = mlperf_prepare(bench_dirpath, dataset_dirpath, 'v4.1', direct)
        prepared = True

    if version == 'mlperf_v0.5':
        # One Get MLPerfv0.5 ONNXs or Instances or Both
        instances = mlperf_prepare(bench_dirpath, dataset_dirpath, 'v0.5', direct)
        prepared = True

    if prepared:
        logger.info(f' = Extracted Dataset From Benchmark. Dataset Size: {len(instances)}')
