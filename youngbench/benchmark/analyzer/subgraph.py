#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-19 14:01
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import networkx
import itertools

from typing import Any, Set, List, Generator


def get_connected_subgraphs(size: int, graph: networkx.Graph) -> Set[Set[Any]]:
    # Karakashian, Shant Kirakos et al. “An Algorithm for Generating All Connected Subgraphs with k Vertices of a Graph.” (2013).
    #assert isinstance(graph, networkx.Graph)
    copy_graph = graph.__class__()
    copy_graph.add_nodes_from(graph.nodes)
    copy_graph.add_edges_from(graph.edges)
    graph = copy_graph
    vertices = list()
    for node in graph.nodes:
        vertices.append(node)

    all_combinations = set()
    for vertex in vertices:
        combinations = get_combinations_with_v(vertex, size, graph)
        all_combinations |= combinations
        graph.remove_node(vertex)
    return all_combinations


def get_combinations_with_v(vertex: Any, size: int, graph: networkx.Graph) -> Set[Set[Any]]:
    combination_tree = networkx.DiGraph()

    def get_combination_tree() -> None:
        node = len(combination_tree)
        relatives = {vertex, }
        combination_tree.add_node(node, relatives=relatives, vertex=vertex, seen=False)

        vertex_visited = dict()
        def build_tree(current_node: int, depth: int) -> networkx.DiGraph:
            if depth == size:
                return
            current_vertex = combination_tree.nodes[current_node]['vertex']
            child_relatives = set(combination_tree.nodes[current_node]['relatives'])
            for child_vertex in graph[current_vertex]:
                if child_vertex != current_vertex and child_vertex not in child_relatives:
                    child_node = len(combination_tree)
                    child_relatives.add(child_vertex)
                    combination_tree.add_node(child_node, relatives=child_relatives, vertex=child_vertex, new=False)
                    combination_tree.add_edge(current_node, child_node)
                    if not vertex_visited.get(child_vertex, False):
                        vertex_visited[child_vertex] = True
                        combination_tree.nodes[child_node]['new'] = True
                    build_tree(child_node, depth+1)
        build_tree(0, 1)
        return

    def get_combinations_from_tree(root: int, size: int) -> Set[Set[int]]:
        node_sets = set()
        if size == 1:
            node_sets.add(frozenset({root}))
        for i in range(1, min(len(combination_tree[root]), size-1)+1):
            for node_combination in get_node_combinations(combination_tree[root], i):
                for size_composition in get_size_compositions(size-1, i):
                    all_subtree_node_sets = list()
                    for position in range(i):
                        subtree_root = node_combination[position]
                        subtree_size = size_composition[position]
                        subtree_node_sets = get_combinations_from_tree(subtree_root, subtree_size)
                        all_subtree_node_sets.append(subtree_node_sets)
                        if len(all_subtree_node_sets[-1]) == 0:
                            break
                    if len(all_subtree_node_sets[-1]) == 0:
                        continue
                    for node_set in overall_union_product(all_subtree_node_sets):
                        node_sets.add(frozenset(node_set | {root}))
        return node_sets

    def overall_union_product(all_node_sets: List[Set[Set[int]]]) -> Set[Set[int]]:
        overall_product_sets = all_node_sets[0]
        for node_sets in all_node_sets[1:]:
            overall_product_sets = union_product(overall_product_sets, node_sets)
        return overall_product_sets

    def union_product(b_s1: Set[Set[int]], b_s2: Set[Set[int]]) -> Set[Set[int]]:
        if not len(b_s1):
            return frozenset()
        if not len(b_s2):
            return b_s1
        
        product_set = set()
        for s_s1 in b_s1:
            s_s1_vertices = {combination_tree.nodes[i]['vertex'] for i in s_s1}
            for s_s2 in b_s2:
                s_s2_vertices = {combination_tree.nodes[j]['vertex'] for j in s_s2}
                valid = False
                if not len(s_s1_vertices & s_s2_vertices):
                    if any(combination_tree.nodes[j]['new'] for j in s_s2):
                        valid = True
                    
                    if all(not len(s_s2_vertices & {combination_tree.nodes[l]['vertex'] for l in combination_tree[i]}) for i in s_s1):
                        valid = True
                if valid:
                    product_set.add(s_s1 | s_s2)
        return product_set

    get_combination_tree()

    combinations = set()
    for combination_from_tree in get_combinations_from_tree(0, size):
        combination = set()
        for node in combination_from_tree:
            combination.add(combination_tree.nodes[node]['vertex'])
        combinations.add(frozenset(combination))
    return combinations


def get_node_combinations(nodes: List[int], number: int) -> Generator[List[int], None, None]:
    return itertools.combinations(nodes, number)


def get_size_compositions(size: int, number: int) -> Generator[List[int], None, None]:
    if size < 0 or number < 0:
        return
    if number == 0:
        if size == 0:
            yield list()
        return
    if number == 1:
        yield [size, ]
        return

    for i in range(1, size):
        for size_composition in get_size_compositions(size-i, number-1):
            yield [i,] + size_composition
    
    return