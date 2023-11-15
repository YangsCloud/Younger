#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-14 16:30
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, List, Dict, Union, Tuple

from youngbench.benchmark.analyzer.network import get_networks, get_networks_have_model, get_networks_with_subnetworks
from youngbench.dataset.modules import Dataset, Prototype
from youngbench.constants import ONNXOperandType


def get_egstats_of_prototype(prototype: Prototype) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], int]:
    # dict{
    #   tuple(
    #       tuple(op_type, op_domain),
    #       tuple(op_type, op_domain)
    #   ): num: int
    # }
    egstats = dict()
    for (u_nid, v_nid), edge in prototype.nn_graph.edges.items():
        u_node = prototype.nn_graph.nodes[u_nid]
        v_node = prototype.nn_graph.nodes[v_nid]
        u_op = (u_node['type'], u_node['domain'])
        v_op = (v_node['type'], v_node['domain'])
        eg = (u_op, v_op)
        egstat = egstats.get(eg, 0)
        egstat += 1
        egstats[eg] = egstat

    return egstats


def get_egstats_of_dataset(dataset: Dataset, count_model: bool = True) -> Dict[str, int]:
    # dict{
    #   tuple(
    #       tuple(op_type, op_domain),
    #       tuple(op_type, op_domain)
    #   ): num: int
    # }
    all_networks = get_networks(dataset)
    egstats = dict()
    for network in all_networks.values():
        egstats_of_net = get_egstats_of_prototype(network)
        for eg, egstat_of_pt in egstats_of_net.items():
            egstat = egstats.get(eg, 0)
            egstat += egstat_of_pt * (max(1, len(network.models)) if count_model else 1)
            egstats[eg] = egstat

    return egstats