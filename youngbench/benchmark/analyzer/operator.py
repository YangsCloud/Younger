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


import onnx

from typing import Set, Dict, Tuple

from youngbench.dataset.modules import Dataset
from youngbench.constants import ONNX


def get_operator_statistics(dataset: Dataset) -> Dict[Tuple[str, str], int]:
    # All Op and Each Op Num
    operator_statistics = dict()
    for instance_identifier, instance in dataset.instances.items():
        for nn_node_id, nn_node in instance.network.nn_nodes.items():
            statistics = operator_statistics.get((nn_node.type, nn_node.domain), dict(number=0, nodes=set()))
            statistics['number'] += 1
            statistics['nodes'].add(nn_node)
            node = network.prototype.get_node(node_id)
            op = (node['op_type'], node['op_domain'])
            op_num = dataset_ops.get(op, 0)
            dataset_ops[op] = op_num + 1

    return dataset_ops


def get_op_stats(dataset: Dataset, op_type: str, op_domain: str) -> Dict:
    # Op Num in Each Network
    op = (op_type, op_domain)
    op_per_network = dict()
    for network in dataset.networks:
        network_id = network.identifier
        num = op_per_network.get(network_id, 0)
        for node_id in network.prototype.node_ids:
            node = network.prototype.get_node(node_id)
            if op == (node['op_type'], node['op_domain']):
                num += 1

        op_per_network[network_id] = num

    return op_per_network