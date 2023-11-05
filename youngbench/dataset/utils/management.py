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

from youngbench.dataset.modules import Model, Prototype, Network, Instance, Dataset


def is_model_in_dataset(model: Model, dataset: Dataset) -> bool:
    network = Network.extract_network(model=model)
    instance = dataset.instances.get(network.prototype.identifier, None)
    if instance is None:
        flag = False
    else:
        network = instance.networks.get(network.identifier, None)
        if network is None:
            flag = False
        else:
            flag = model.identifier in network.models.keys()
    return flag


def add_model_to_dataset(model: Model, dataset: Dataset) -> bool:
    network = Network.extract_network(model)
    network.insert(model)

    instance = Instance()
    instance.setup_prototype(network.prototype)
    instance.insert(network)

    if dataset.insert(instance):
        logger.info(f' [YBD] -> Model Insertion Successful, model: {model.identifier}.')
        flag = True
    else:
        logger.info(f' [YBD] -> Skip, model exists: {model.identifier}.')
        flag = False
    return flag


def is_network_in_dataset(network: Network, dataset: Dataset) -> bool:
    instance = dataset.instances.get(network.prototype.identifier, None)
    if instance is None:
        flag = False
    else:
        flag = network.identifier in instance.networks.keys()
    return flag


def add_network_to_dataset(network: Network, dataset: Dataset) -> bool:
    instance = Instance()
    instance.setup_prototype(network.prototype)
    instance.insert(network)
    if dataset.insert(instance):
        logger.info(f' [YBD] -> Network Insertion Successful, network: {network.identifier}.')
        flag = True
    else:
        logger.info(f' [YBD] -> Skip, network exists: {network.identifier}.')
        flag = False
    return flag


def is_prototype_in_dataset(prototype: Prototype, dataset: Dataset) -> bool:
    instance = dataset.instances.get(prototype.identifier, None)
    if instance is None:
        flag = False
    else:
        flag = True
    return flag


def add_prototype_to_dataset(prototype: Prototype, dataset: Dataset) -> bool:
    instance = Instance()
    instance.setup_prototype(prototype)
    if dataset.insert(instance):
        logger.info(f' [YBD] -> Prototype Insertion Successful, prototype: {prototype.identifier}.')
        flag = True
    else:
        logger.info(f' [YBD] -> Skip, prototype exists: {prototype.identifier}.')
        flag = False
    return flag


def enrich_dataset(onnx_model: onnx.ModelProto, dataset: Dataset) -> bool:
    flag = True

    model = Model(onnx_model=onnx_model)
    network, deep_networks = Network.extract_networks(model, deep_extract=True)
    flag &= add_network_to_dataset(network, dataset)

    for deep_network in deep_networks:
        flag &= add_network_to_dataset(deep_network, dataset)

    flag &= add_model_to_dataset(model, dataset)

    return flag