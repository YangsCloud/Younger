#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-10 10:17
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, List, Dict, Union, Tuple

from youngbench.benchmark.analyzer.network import get_networks, get_networks_have_model, get_networks_with_subnetworks
from youngbench.dataset.modules import Dataset, Prototype
from youngbench.constants import ONNXOperandType


def get_opstats_of_prototype(prototype: Prototype) -> Dict[Tuple[str, str], Dict[str, Union[int, bool]]]:
    # dict{
    #   tuple(op_type, op_domain): dict{num: int, cus: bool}
    # }
    opstats = dict()
    for node in prototype.nn_graph.nodes.values():
        op = (node['type'], node['domain'])
        opstat = opstats.get(op, dict(num=0, cus=False))
        opstat['num'] += 1
        opstat['cus'] |= bool(node['is_custom'])
        opstats[op] = opstat

    return opstats


def get_opstats_of_dataset(dataset: Dataset, count_model: bool = True) -> Dict[Tuple[str, str], Dict[str, Union[int, bool]]]:
    # dict{
    #   tuple(op_type, op_domain): dict{num: int, cus: bool}
    # }
    all_networks = get_networks(dataset)
    opstats = dict()
    for network in all_networks.values():
        opstats_of_net = get_opstats_of_prototype(network)
        for op, opstat_of_pt in opstats_of_net.items():
            opstat = opstats.get(op, dict(num=0, cus=False))
            opstat['num'] += opstat_of_pt['num'] * (len(network.models) if count_model else 1)
            opstat['cus'] |= opstat_of_pt['cus']
            opstats[op] = opstat

    return opstats


def get_opstats_per_model(dataset: Dataset) -> Dict[str, Dict[Tuple[str, str], Dict[str, Union[int, bool]]]]:
    # dict{
    #   model_identifier: dict{
    #     tuple(op_type, op_domain): dict{num: int, cus: bool}
    #   }
    # }
    networks_have_model = get_networks_have_model(dataset)
    networks_with_subnetworks = get_networks_with_subnetworks(dataset)
    opstats = dict()
    for network_id, subnetworks in networks_with_subnetworks.items():
        opstats_of_net = get_opstats_of_prototype(networks_have_model[network_id])
        for subnetwork in subnetworks:
            opstats_of_subnet = get_opstats_of_prototype(subnetwork)
            for op, opstat_of_subnet in opstats_of_subnet.items():
                opstat_of_net = opstats_of_net.get(op, dict(num=0, cus=False))
                opstat_of_net['num'] += opstat_of_subnet['num']
                opstat_of_net['cus'] |= opstat_of_subnet['cus']
                opstats_of_net[op] = opstat_of_net

        opstats[network_id] = opstats_of_net

    return opstats