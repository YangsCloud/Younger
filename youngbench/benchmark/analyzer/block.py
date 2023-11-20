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


import networkx

from typing import List, Dict, Tuple

from youngbench.benchmark.analyzer.subgraph import shallow_copy_of_graph, get_weakly_connected_subgraphs

from youngbench.dataset.modules import Dataset, Prototype


def get_blocks_of_prototype(prototype: Prototype, block_sizes: List[int], max_ind: int = 15, max_outd: int = 15) -> List[Dict[str, Tuple[Prototype, ]]]:
    # A block of the `prototype` is a subgraph (meet some constraints below) of the graph represented by prototype.nn_graph (networkx.DiGraph),
    # Constraints:
    #  1. A valid block must have only one input node (in_degree == 0) and one output node (out_degree == 0);
    #  2. All middle nodes must have the same inputs with the original `graph`.
    prototype_graph = prototype.nn_graph
    directed_graph = shallow_copy_of_graph(prototype_graph, node_attribute=True)

    for nid, node in prototype_graph.nodes.items():
        if prototype_graph.in_degree(nid) > max_ind or prototype_graph.out_degree(nid) > max_outd or node['is_custom'] or node['has_subgraph']:
            directed_graph.remove_node(nid)

    blocks = list()
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
                blocks_at_size.append(weakly_connected_subgraph.nodes(data=True))
        blocks.append(blocks_at_size)
    
    return blocks