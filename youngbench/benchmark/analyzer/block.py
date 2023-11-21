#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-10 11:47
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import re
import pathlib
import networkx

from typing import List, Dict, Tuple, Union

from youngbench.benchmark.analyzer.network import get_networks
from youngbench.benchmark.analyzer.subgraph import shallow_copy_of_graph, get_weakly_connected_subgraphs

from youngbench.dataset.modules import Dataset, Prototype


def get_abstract_block(block: networkx.DiGraph) -> networkx.DiGraph:
    abstract_block = shallow_copy_of_graph(block)
    for nid in abstract_block.nodes:
        abstract_block.nodes[nid].update(op=[block.nodes[nid]['type'], block.nodes[nid]['domain']])
    return abstract_block


def get_weisfeiler_lehman_hash_of_block(block: networkx.DiGraph) -> str:
    wl_hash = networkx.weisfeiler_lehman_graph_hash(block, node_attr='op')
    return wl_hash


def get_bkstats_of_prototype(prototype: Prototype, block_sizes: List[int], max_ind: int = 15, max_outd: int = 15) -> Dict[int, List[networkx.DiGraph]]:
    # A block of the `prototype` is a subgraph (meet some constraints below) of the graph represented by prototype.nn_graph (networkx.DiGraph),
    # Constraints:
    #  1. A valid block must have only one input node (in_degree == 0) and one output node (out_degree == 0);
    #  2. All middle nodes must have the same inputs with the original `graph`.
    # 
    # dict{
    #   int(block_size): list[networkx.DiGraph]
    # }
    prototype_graph = prototype.nn_graph
    directed_graph = shallow_copy_of_graph(prototype_graph, node_attribute=True)

    for nid, node in prototype_graph.nodes.items():
        if prototype_graph.in_degree(nid) > max_ind or prototype_graph.out_degree(nid) > max_outd or node['is_custom'] or node['has_subgraph']:
            directed_graph.remove_node(nid)

    blocks = dict()
    for block_size in block_sizes:
        blocks_at_size = list()
        weakly_connected_subgraphs = get_weakly_connected_subgraphs(block_size, directed_graph, detail=True)
        for weakly_connected_subgraph in weakly_connected_subgraphs:
            valid = True
            input_node = None
            output_node = None
            for node in weakly_connected_subgraph.nodes:
                ind = weakly_connected_subgraph.in_degree(node)
                outd = weakly_connected_subgraph.out_degree(node)
                if ind != 0 and outd != 0:
                    # Constraint 2
                    if ind != prototype_graph.in_degree(node):
                        valid = False
                        break
                else:
                    # Constraint 1
                    if ind == 0:
                        if input_node is None:
                            input_node = node
                        else:
                            valid = False
                            break
                    if outd == 0:
                        if output_node is None:
                            output_node = node
                        else:
                            valid = False
                            break
            if valid:
                blocks_at_size.append(weakly_connected_subgraph)
        blocks[block_size] = blocks_at_size

    return blocks


def get_bkstats_of_dataset(dataset: Dataset, block_sizes: List[int], max_ind: int = 15, max_outd: int = 15, count_model: bool = True) -> Dict[int, Dict[str, Dict[str, Union[int, networkx.DiGraph, List[Tuple[str, networkx.DiGraph]]]]]]:
    # dict{
    #   int(block_size): dict{
    #     wl_hash: dict{
    #       num: int,
    #       pool: list[(network_identifier: str, block_at_size: networkx.DiGraph)]
    #       absbk: networkx.DiGraph
    #     }
    #   }
    # }
    # wl_hash is weisfeiler_lehman_graph_hash, isomorphic graphs have the same hash value.

    all_networks = get_networks(dataset)
    bkstats = dict()
    for netid, network in all_networks.items():
        bkstats_of_net = get_bkstats_of_prototype(network, block_sizes, max_ind=max_ind, max_outd=max_outd)
        for block_size, blocks_at_size in bkstats_of_net.items():
            bkstat = bkstats.get(block_size, dict())

            network_identifier = network.identifier
            for block_at_size in blocks_at_size:
                abstract_block = get_abstract_block(block_at_size)
                wl_hash = get_weisfeiler_lehman_hash_of_block(abstract_block)
                iso_bkstat = bkstat.get(wl_hash, dict(num=0, pool=list(), absbk=abstract_block)) # Isomorphic Block

                iso_bkstat['num'] += 1 * (max(1, len(network.models)) if count_model else 1)
                iso_bkstat['pool'].append((network_identifier, block_at_size))

                bkstat[wl_hash] = iso_bkstat

            bkstats[block_size] = bkstat

    return bkstats


def save_bkstats_of_dataset(bkstats_of_dataset, dirpath):
    for block_size, bkstats_of_dataset_at_size in bkstats_of_dataset.items():
        for wl_hash, bkstat in bkstats_of_dataset_at_size.items():
            save_dir = dirpath.joinpath(f'bk_{block_size}-wl_{wl_hash}-num_{bkstat["num"]}')
            save_dir.mkdir(parents=True, exist_ok=True)

            networkx.write_gml(bkstat['absbk'], save_dir.joinpath(f'absbk.gml'))
            for index, (netid, block) in enumerate(bkstat['pool']):
                networkx.write_gml(block, save_dir.joinpath(f'id_{index}-ni_{netid}.gml'))


def load_bkstats_of_dataset(dirpath: pathlib.Path):
    bkstats_of_dataset = dict()
    bk_pattern = r'bk_(?P<block_size>\d+)-wl_(?P<wl_hash>\w+)-num_(?P<num>\d+)'
    pl_pattern = r'id_(?P<index>\d+)-ni_(?P<netid>\w+).gml'
    for bkstat_dirpath in dirpath.iterdir():
        if bkstat_dirpath.is_dir():
            bk_match = re.match(bk_pattern, bkstat_dirpath.name)
            if bk_match:
                block_size = int(bk_match.group('block_size'))
                wl_hash = bk_match.group('wl_hash')
                num = int(bk_match.group('num'))
                pool = list()
                absbk = networkx.read_gml(bkstat_dirpath.joinpath('absbk.gml'))

                assert get_weisfeiler_lehman_hash_of_block(absbk) == wl_hash, f'absbk wl_hash: {get_weisfeiler_lehman_hash_of_block(absbk)}; wl_hash: {wl_hash}'

                pd = dict()
                for pl_filepath in bkstat_dirpath.iterdir():
                    pl_match = re.match(pl_pattern, pl_filepath.name)
                    if pl_match:
                        index = int(pl_match.group('index'))
                        netid = pl_match.group('netid')
                        block = networkx.read_gml(pl_filepath)
                        pd[index] = (netid, block)

                for i in range(len(pd)):
                    pool.append(pd[i])

                bkstats_of_dataset_at_size = bkstats_of_dataset.get(block_size, dict())
                bkstats_of_dataset_at_size[wl_hash] = dict()
                bkstats_of_dataset_at_size[wl_hash]['num'] = num
                bkstats_of_dataset_at_size[wl_hash]['absbk'] = absbk
                bkstats_of_dataset_at_size[wl_hash]['pool'] = pool
                bkstats_of_dataset[block_size] = bkstats_of_dataset_at_size

    return bkstats_of_dataset