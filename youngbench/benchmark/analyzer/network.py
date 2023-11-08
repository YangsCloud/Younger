#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-08 16:34
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, List, Dict

from youngbench.dataset.modules import Dataset, Network


def get_networks(dataset: Dataset) -> Dict[str, Network]:
    networks = dict()
    for instance in dataset.instances.values():
        for network_id, network in instance.networks.items():
            networks[network_id] = network

    return networks


def get_networks_have_model(dataset: Dataset) -> Dict[str, Network]:
    networks_with_model = dict()
    for instance in dataset.instances.values():
        for network_id, network in instance.networks.items():
            if len(network.models) == 0:
                continue
            else:
                networks_with_model[network_id] = network

    return networks_with_model


def get_networks_with_subnetwork_ids(dataset: Dataset) -> Dict[str, Set[str]]:
    # Only network with models
    networks = get_networks(dataset)
    networks_have_model = get_networks_have_model(dataset)
    networks_with_subnetwork_ids = dict()

    def dfs(network: Network) -> Set[str]:
        all_subnetwork_id = set()
        for nn_node in network.nn_nodes.values():
            if nn_node.has_subgraph:
                if nn_node.is_custom:
                    subnetwork_id = nn_node.attributes['__YBD_function__']
                    assert isinstance(subnetwork_id, str)
                    all_subnetwork_id.add(subnetwork_id)
                    all_subnetwork_id.update(dfs(networks[subnetwork_id]))
                else:
                    for attribute in nn_node.attributes.values():
                        if isinstance(attribute, list):
                            for subnetwork_id in attribute:
                                assert isinstance(subnetwork_id, str)
                                all_subnetwork_id.add(subnetwork_id)
                                all_subnetwork_id.update(dfs(networks[subnetwork_id]))

        return all_subnetwork_id


    for network_id, network in networks_have_model.items():
        networks_with_subnetwork_ids[network_id] = dfs(network)

    return networks_with_subnetwork_ids


def get_networks_with_subnetworks(dataset: Dataset) -> Dict[str, List[Network]]:
    # Only network with models
    networks = get_networks(dataset)
    networks_with_subnetwork_ids = get_networks_with_subnetwork_ids(dataset)

    networks_with_subnetworks = dict()
    for network_id, subnetwork_ids in networks_with_subnetwork_ids.items():
        networks_with_subnetworks[network_id] = list()
        for subnetwork_id in subnetwork_ids:
            networks_with_subnetworks[network_id].append(networks[subnetwork_id])

    return networks_with_subnetworks