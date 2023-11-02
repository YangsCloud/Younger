#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-01 10:41
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import onnx

from youngbench.logging import logger

from youngbench.dataset.modules import Model, Network, Instance, Dataset
from youngbench.dataset.utils.extraction import extract_network, extract_networks


def is_model_in_dataset(model: Model, dataset: Dataset) -> bool:
    network = extract_network(model=model)
    instance = dataset.instances.get(network.identifier, None)
    if instance is None:
        flag = False
    else:
        flag = model.identifier in instance.models.keys()
    return flag


def add_model_to_dataset(model: Model, dataset: Dataset) -> None:
    network = extract_network(model=model)
    instance = dataset.instances.get(network.identifier, None)
    if instance is not None and model.identifier in instance.models.keys():
        logger.info(f' [YBD] -> Skip, model exists: {model.identifier}.')
    else:
        instance = Instance(network=network, models=[model, ])
        dataset.insert(instance)
        logger.info(f' [YBD] -> Model Insertion Successful, model: {model.identifier}.')
    return


def is_network_in_dataset(network: Network, dataset: Dataset) -> bool:
    if network.identifier in dataset.instances.keys():
        flag = True
    else:
        flag = False
    return flag


def add_network_to_dataset(network: Network, dataset: Dataset) -> None:
    if is_network_in_dataset(network, dataset):
        logger.info(f' [YBD] -> Skip, network exists: {network.identifier}.')
    else:
        instance = Instance(network=network)
        dataset.insert(instance)
        logger.info(f' [YBD] -> Network Insertion Successful, network: {network.identifier}.')
    return


def enrich_dataset(onnx_model: onnx.ModelProto, dataset: Dataset) -> None:
    model = Model(onnx_model=onnx_model)
    network, deep_networks = extract_networks(model, deep_extract=True)

    add_network_to_dataset(network, dataset)
    add_model_to_dataset(model, dataset)
    for deep_network in deep_networks:
        add_network_to_dataset(deep_network, dataset)
    return