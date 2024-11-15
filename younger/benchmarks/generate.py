#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-07-15 15:46
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import toml
import numpy
import pathlib

import hdbscan
import sklearn.cluster

from typing import Any, Literal

from younger.commons.io import load_pickle, load_json
from younger.commons.logging import logger


support_clt_modes = set([
    'HDBSCAN',
    'KMeans',
])

support_platforms = set([
    'GPU',
    'CPU',
    'Edge',
])


support_prioritys = set([
    'OP',
    'CT',
])


support_providers = set([
    'CPU',
    'OpenVINO',
    'CUDA',
    'TensorRT',
    'NNAPI',
    'QNN',
    'XNNPACK',
    'CoreML',
])


def select_graphs(graphs: dict[str, int | set[str] | str], want_opset: set[str], must_opset: set[str], cluster_method: Literal['HDBSCAN', 'KMeans'], cluster_config: dict[str, Any], priority: Literal['OP', 'CT']):

    daglabs = [daglab for daglab, dagemb in sorted(list(dagembs.items()))]
    dagembs = [dagemb for daglab, dagemb in sorted(list(dagembs.items()))]

    if cluster_method == 'HDBSCAN':
        cluster = hdbscan.HDBSCAN(**cluster_config)
        dagclts = cluster.fit_predict(numpy.array(dagembs))
        cluster.cluster_

    dagclts = 

    graphs = dict()
    clt2grps = dict()
    for graph_hash, op_detail in result['op_details'].items():
        graph_opset = set(op_detail.keys())
        if (not user_op_loose and len(graph_opset - want_opset) != 0) or len(graph_opset - must_opset) != 0:
            continue
        # (Cost, Operator Set, Cluster, Parent Graph Hash)
        # Maybe the Cost will be set in the future.
        graphs[graph_hash] = dict(
            cost = 1,
            opset = graph_opset,
            cluster = dagclts[graph_hash],
            identifier = result['s2p_hash'][graph_hash]
        )


def main(benchmark_dirpath: pathlib.Path, configuration_filepath: pathlib.Path):
    configuration = toml.load(configuration_filepath)

    # TODO: Can specify platform (ONNX Execution Provider) which contain specific opeartor set.
    user_requirements: dict = configuration.get('user_requirements', dict())
    user_clt_mode: str = user_requirements.get('clt_mode', 'HDBSCAN')
    user_clt_args: str = user_requirements.get('clt_args', dict())
    user_platform: str = user_requirements.get('platform', 'CPU')
    user_provider: str = user_requirements.get('provider', 'CPU')
    user_priority: str = user_requirements.get('priority', 'OP')
    user_op_focus: str = user_requirements.get('op_focus', None)
    user_op_loose: str = user_requirements.get('op_loose', True)

    # For more details about Execution Providers: https://onnxruntime.ai/docs/execution-providers/
    assert user_clt_mode in support_clt_modes, f'Only Support Cluster Method: KMeans, HDBSCAN! Yours Is {user_clt_mode}'
    assert user_platform in support_platforms, f'Only Support Platform: GPU, CPU, and Edge! Yours Is {user_platform}'
    assert user_priority in support_prioritys, f'Only Support Generation Cover Priority: OP (Operator) and CT (Cluster)! Yours Is {user_priority}'
    assert user_provider in support_providers, f'This Execution Provider - {user_provider} - Is Not Supportted Yet!'

    if user_op_focus:
        user_opset = set(load_json(user_op_focus))
    else:
        user_opset = set()

    system_requirements: dict = configuration['system_requirements']

    providers = load_json(system_requirements['providers_filepath']) # Execution Provider Supportted Operator Sets
    provider_opset = set(providers[user_provider][user_platform])
    result = load_pickle(system_requirements['result_filepath'])
    younger_opset = [oplab for oplab, opemb in result['opembs'].items()]

    must_opset = provider_opset - ( provider_opset - younger_opset )
    logger.info(f'The following Operator in EP {user_provider} ({user_platform}) are not contained in Younger: {provider_opset - younger_opset}')
    want_opset = user_opset - ( user_opset - must_opset )
    logger.info(f'The following Operator that user specified are not contained in EP {user_provider} ({user_platform}) or Younger: {user_opset - must_opset}')

    assert len(result['op_details']) == len(result['dagembs'])

    graphs: dict[str, int | set[str] | str] = dict()
    for graph_hash, op_detail in result['op_details'].items():
        graph_opset = set(op_detail.keys())
        if (not user_op_loose and len(graph_opset - want_opset) != 0) or len(graph_opset - must_opset) != 0:
            continue
        # (Cost, Operator Set, Cluster, Parent Graph Hash)
        # Maybe the Cost will be set in the future.
        graphs[graph_hash] = dict(
            cost = 1,
            opset = graph_opset,
            parent = result['s2p_hash'][graph_hash]
        )

    graphs = select_graphs(graphs, want_opset, must_opset, user_clt_mode, user_clt_args, user_priority)

    hash2names = load_json(system_requirements['hash2names_filepath'])
