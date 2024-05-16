#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-16 23:43
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import time
import torch
import pathlib

from typing import Literal

from torch_geometric.data import Batch, Data

from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.constructors.official.filter import complete_attributes_of_node

from younger.applications.utils.neural_network import get_model_parameters_number, get_device_descriptor, load_checkpoint
from younger.applications.performance_prediction.models import NAPPGATVaryV1
from younger.applications.performance_prediction.datasets import YoungerDataset


def main(
    meta_filepath: pathlib.Path,
    checkpoint_filepath: pathlib.Path,
    onnx_models_dirpath: pathlib.Path,
    max_inclusive_version: int,

    node_dim: int = 512,
    task_dim: int = 512,
    hidden_dim: int = 512,
    readout_dim: int = 256,
    cluster_num: int | None = None,

    device: Literal['CPU', 'GPU'] = 'GPU',
):
    assert device in {'CPU', 'GPU'}
    device_descriptor = get_device_descriptor(device, 0)
    assert torch.cuda.is_available() or device == 'CPU'

    logger.info(f'Using Device: {device};')

    logger.info(f'  v Loading Meta ...')
    meta = YoungerDataset.load_meta(meta_filepath)
    logger.info(f'    -> Tasks Dict Size: {len(meta["t2i"])}')
    logger.info(f'    -> Nodes Dict Size: {len(meta["n2i"])}')
    logger.info(f'  ^ Built.')

    logger.info(f'  v Building Younger Model ...')
    model = NAPPGATVaryV1(
        meta=meta,
        node_dim=node_dim,
        task_dim=task_dim,
        hidden_dim=hidden_dim,
        readout_dim=readout_dim,
        cluster_num=cluster_num,
        mode='Supervised'
    )

    parameters_number = get_model_parameters_number(model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    logger.info(
        f'\n  - Model Architecture:'
        f'\n{model}'
        f'\n  - Number of Parameters:'
        f'\n{parameters_number_str}'
        f'\n  ^ Built.'
    )

    logger.info(f'  v Loading Model Weights From Checkpoint [\'{checkpoint_filepath}\']...')
    checkpoint = load_checkpoint(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    logger.info(f'  ^ Loaded ')

    logger.info(f'  v Moving model to the specified device ...')
    model.to(device_descriptor)
    logger.info(f'  ^ Moved.')

    logger.info(f'  v Loading ONNX Models')
    datas = list()
    onnx_model_filenames = list()
    for onnx_model_filepath in onnx_models_dirpath.iterdir():
        onnx_model_filenames.append(onnx_model_filepath.name)
        instance = Instance(onnx_model_filepath)
        standardized_graph = Network.standardize(instance.network.graph)
        for node_index in standardized_graph.nodes():
            standardized_graph[node_index]['features'] = complete_attributes_of_node(standardized_graph[node_index]['features'], max_inclusive_version)
        standardized_graph.graph.clear()
        data = YoungerDataset.get_data(standardized_graph, meta, feature_get_type='none')
        datas.append(data)
    batch = Batch.from_data_list(datas)
    logger.info(f'  ^ Loaded. Total - {len(datas)}.')

    model.eval()
    logger.info(f'  -> Interact Test Begin ...')
    tic = time.time()

    results = list()
    with torch.no_grad():
        batch: Data = batch.to(device_descriptor)
        output, _ = model(batch.x, batch.edge_index, batch.batch)

        for onnx_model_filename, output_value in zip(onnx_model_filenames, output):
            logger.info(f'  -> Result - {onnx_model_filename}: {output_value}')
    toc = time.time()

    logger.info(f'  -> Test Finished. (Time Cost = {toc-tic:.2f}s)')
