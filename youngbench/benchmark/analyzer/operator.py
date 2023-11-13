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


def get_opstats_of_prototype(prototype: Prototype) -> Dict[str, Dict[str, Union[int, bool, Dict[str, str]]]]:
    # dict{
    #   tuple(op_type, op_domain): dict{num: int, cus: bool, degree: {(ind, outd): int}}
    # }
    opstats = dict()
    for nid, node in prototype.nn_graph.nodes.items():
        ind = prototype.nn_graph.in_degree(nid)
        outd = prototype.nn_graph.out_degree(nid)
        xd = str((ind, outd))
        op = str((node['type'], node['domain']))
        opstat = opstats.get(op, dict(num=0, cus=False, degree=dict()))
        opstat['num'] += 1
        opstat['cus'] |= bool(node['is_custom'])
        opstat_degree = opstat['degree'].get(xd, 0)
        opstat_degree += 1
        opstat['degree'][xd] = opstat_degree
        opstats[op] = opstat

    return opstats


def get_opstats_of_dataset(dataset: Dataset, count_model: bool = True) -> Dict[str, Dict[str, Union[int, bool, Dict[str, str]]]]:
    # dict{
    #   tuple(op_type, op_domain): dict{num: int, cus: bool, degree: {(ind, outd): int}}
    # }
    all_networks = get_networks(dataset)
    opstats = dict()
    for network in all_networks.values():
        opstats_of_net = get_opstats_of_prototype(network)
        for op, opstat_of_pt in opstats_of_net.items():
            opstat = opstats.get(op, dict(num=0, cus=False, degree=dict()))
            opstat['num'] += opstat_of_pt['num'] * (max(1, len(network.models)) if count_model else 1)
            opstat['cus'] |= opstat_of_pt['cus']
            for degree, num in opstat_of_pt['degree'].items():
                opstat_degree_of_pt = opstat['degree'].get(degree, 0)
                opstat_degree_of_pt += num * (max(1, len(network.models)) if count_model else 1)
                opstat['degree'][degree] = opstat_degree_of_pt
            opstats[op] = opstat

    return opstats


def get_opstats_per_model(dataset: Dataset) -> Dict[str, Dict[str, Dict[str, Union[int, bool, Dict[str, str]]]]]:
    # dict{
    #   model_identifier: dict{
    #     tuple(op_type, op_domain): dict{num: int, cus: bool, degree: {(ind, outd): int}}
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
                opstat_of_net = opstats_of_net.get(op, dict(num=0, cus=False, degree=dict()))
                opstat_of_net['num'] += opstat_of_subnet['num']
                opstat_of_net['cus'] |= opstat_of_subnet['cus']
                for degree, num in opstat_of_subnet['degree'].items():
                    opstat_degree_of_net = opstat_of_net['degree'].get(degree, 0)
                    opstat_degree_of_net += num
                    opstat_of_net['degree'][degree] = opstat_degree_of_net
                opstats_of_net[op] = opstat_of_net

        opstats[network_id] = opstats_of_net

    return opstats


def get_opstats_of_xput(dataset: Dataset) -> Dict[str, Dict[str, int]]:
    # dict{
    #   input/output: dict{
    #     tuple(op_type, op_domain): dict{(input_num, output_num): int}
    #   }
    # }
    networks_have_model = get_networks_have_model(dataset)

    opstats = dict(input=dict(), output=dict())
    for network in networks_have_model.values():
        for nid, node in network.nn_graph.nodes.items():
            op = str((node['type'], node['domain']))
            ind = network.nn_graph.in_degree(nid)
            outd = network.nn_graph.out_degree(nid)
            if ind == 0 or outd == 0:
                if ind == 0:
                    kind = 'input'
                if outd == 0:
                    kind = 'output'
                op_input_num = len(network.nn_nodes[str(nid)].operands)
                op_output_num = len(network.nn_nodes[str(nid)].results)
                opstat = opstats[kind].get(op, dict())
                op_xput_num = str((op_input_num, op_output_num))
                xput_stat = opstat.get(op_xput_num, 0)
                xput_stat += 1
                opstat[op_xput_num] = xput_stat
                opstats[kind][op] = opstat

    return opstats