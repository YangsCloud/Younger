#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-15 15:14
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy
import networkx

from typing import List, Dict, Tuple, Union

from youngbench.dataset.modules import Dataset

from youngbench.benchmark.analyzer import get_onnx_operators, get_egstats_of_dataset, get_opstats_of_dataset


def get_op_freqs(dataset: Dataset) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], Dict[int, int]], Dict[Tuple[str, str], Dict[int, int]]]:
    op_freqs = dict()

    op_ind_freqs = dict()
    op_outd_freqs = dict()

    opstats_of_dataset = get_opstats_of_dataset(dataset)
    for op, opstat in opstats_of_dataset.items():
        if not opstat['cus']:
            op_freqs[op] = opstat['num']

            op_ind_occur = dict()
            op_outd_occur = dict()
            for (op_ind, op_outd), num in opstat['degree'].items():
                if op_ind != 0:
                    ind_num = op_ind_occur.get(op_ind, 0)
                    ind_num += num
                    op_ind_occur[op_ind] = ind_num

                if op_outd != 0:
                    outd_num = op_outd_occur.get(op_outd, 0)
                    outd_num += num
                    op_outd_occur[op_outd] = outd_num

            op_ind_freqs[op] = op_ind_occur
            op_outd_freqs[op] = op_outd_occur

    return (op_freqs, op_ind_freqs, op_outd_freqs)


def get_eg_freqs(dataset: Dataset) -> Tuple[
    Dict[Tuple[str, str],
         Dict[Tuple[str, str],
              Union[int, Dict[str, Union[int, str]]]]
        ], 
    Dict[Tuple[str, str],
         Dict[Tuple[str, str],
              Union[int, Dict[str, Union[int, str]]]]
        ]
    ]:
    egstats_of_dataset = get_egstats_of_dataset(dataset)
    eg_freqs = dict()
    eg_rvs_freqs = dict()
    for (u_op, v_op), egstat_of_dataset in egstats_of_dataset.items():
        eg_freq = eg_freqs.get(u_op, dict())
        eg_freq[v_op] = egstat_of_dataset
        eg_freqs[u_op] = eg_freq

        eg_rvs_freq = eg_rvs_freqs.get(v_op, dict())
        eg_rvs_freq[u_op] = egstat_of_dataset
        eg_rvs_freqs[u_op] = eg_rvs_freq
    
    return (eg_freqs, eg_rvs_freqs)


def get_prototype_draft(
        prototype_size: int,
        op_freqs: Dict[Tuple[str, str], int],
        op_outd_freqs: Dict[Tuple[str, str], Dict[int, int]],
        eg_freqs: Dict[Tuple[str, str], Dict[Tuple[str, str], Union[int, Dict[str, Union[int, str]]]]]
    ) -> networkx.DiGraph:
    # Prototype Draft: DAG - the out degree of some nodes may be 0.
    # The Prototype Draft should be fixed.
    assert prototype_size >= 2
    ops = list()
    freqs = list()
    for op, freq in op_freqs.items():
        if op in eg_freqs:
            ops.append(op)
            freqs.append(freq)

    op_indices = numpy.random.choice(len(ops), size=prototype_size, replace=True, p=numpy.array(freqs)/sum(freqs))

    def get_v_nids(u_nid: int) -> Tuple[List[Tuple[str, str]], List[int], List[float]]:
        valid_v_nids = list()
        valid_v_nids_p = list()

        u_op = ops[op_indices[u_nid]]
        possible_v_ops = set(eg_freqs[u_op].keys())
        for v_nid in range(u_nid+1, prototype_size):
            v_op = ops[op_indices[v_nid]]
            if v_op in possible_v_ops:
                valid_v_nids.append(v_nid)
                valid_v_nids_p.append(eg_freqs[u_op][v_op]['num'])
        
        valid_v_nids_p = numpy.array(valid_v_nids_p)/sum(valid_v_nids_p)
        return valid_v_nids, valid_v_nids_p

    prototype_draft = networkx.DiGraph()

    for u_nid in range(prototype_size-1):
        u_op = ops[op_indices[u_nid]]
        prototype_draft.add_node(str(u_nid), type=u_op[0], domain=u_op[1])

        valid_v_nids, valid_v_nids_p = get_v_nids(u_nid)

        u_op_outds = list(op_outd_freqs[u_op].keys())
        u_op_outds_p = numpy.array(list(op_outd_freqs[u_op].values()))/sum(op_outd_freqs[u_op].values())
        u_op_outd = numpy.random.choice(u_op_outds, p=u_op_outds_p)

        if len(valid_v_nids) != 0:
            u_op_outd = min(len(valid_v_nids), u_op_outd)
            v_nids = numpy.random.choice(valid_v_nids, size=u_op_outd, replace=False, p=valid_v_nids_p)
            for v_nid in v_nids:
                v_op = ops[op_indices[v_nid]]
                i = numpy.random.choice(len(eg_freqs[u_op][v_op]['nis']))
                eg_nis = eg_freqs[u_op][v_op]['nis'][i]
                prototype_draft.add_edge(str(u_nid), str(v_nid), **eg_nis)
    
    return prototype_draft


def get_prototype(prototype_size: int, dataset: Dataset) -> networkx.DiGraph:
    assert prototype_size >= 2
    onnx_operators = get_onnx_operators()

    op_freqs, op_ind_freqs, op_outd_freqs = get_op_freqs(dataset)
    eg_freqs, eg_rvs_freqs = get_eg_freqs(dataset)

    prototype = get_prototype_draft(prototype_size, op_freqs, op_outd_freqs, eg_freqs)
    # prototype = fix_prototype(prototype_draft, op_ind_freqs, eg_rvs_freqs)

    assert networkx.is_directed_acyclic_graph(prototype)
    return prototype