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
import semantic_version

from typing import List

from youngbench.logging import logger, logging_level

from youngbench.dataset.modules import Model, Prototype, Network, Instance, Dataset


def check_dataset(dataset: Dataset, whole_check: bool = True) -> None:
    dataset.check()

    def check(acquired_dataset: Dataset) -> None:
        dataset_stamp = max(acquired_dataset.stamps)
        assert dataset_stamp.checksum == acquired_dataset.checksum, (
            f'The \"Checksum={acquired_dataset.checksum}\" of \"Dataset\" (Version={dataset_stamp.version}) does not match \"Stamp={dataset_stamp.checksum}\"'
        )
        for acquired_instance_identifier in acquired_dataset.uniques:
            acquired_instance = acquired_dataset.instances[acquired_instance_identifier]
            instance_stamp = max(acquired_instance.stamps)
            assert instance_stamp.checksum == acquired_instance.checksum, (
                f'The \"Checksum={acquired_instance.checksum}\" of \"Instance\" (Version={instance_stamp.version}) does not match \"Stamp={instance_stamp.checksum}\"'
            )
            for acquired_network_identifier in acquired_instance.uniques:
                acquired_network = acquired_instance.networks[acquired_network_identifier]
                network_stamp = max(acquired_network.stamps)
                assert network_stamp.checksum == acquired_network.checksum, (
                    f'The \"Checksum={acquired_network.checksum}\" of \"Network\" (Version={network_stamp.version}) does not match \"Stamp={network_stamp.checksum}\"'
                )

    if whole_check:
        stamps = sorted(list(dataset.stamps))
        for index, stamp in enumerate(stamps):
            logger.info(f' [YBD] -> No.{index} Checking Dataset Version=[{stamp.version}] ...')
            org_logging_level = logger.level
            mut_logging_level = logging_level['NOTSET']
            logger.setLevel(mut_logging_level)
            acquired_dataset = dataset.acquire(stamp.version)
            logger.setLevel(org_logging_level)
            check(acquired_dataset)
            logger.info(f' [YBD] -> Pass!')
    else:
        logger.info(f' [YBD] -> Checking Dataset Latest Version=[{dataset.latest_version}] ...')
        check(dataset)
        logger.info(f' [YBD] -> Pass!')

    return


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
        logger.info(f' [YBD] -> Network Insertion Successful, network: {network.identifier},')
        logger.info(f' [YBD]                                  details: {network.nn_graph}.')
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