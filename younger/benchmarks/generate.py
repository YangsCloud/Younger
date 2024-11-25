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


import numpy
import pathlib

import hdbscan
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture
import sklearn.neighbors

from typing import Any, Literal
from numpy.typing import NDArray
from KDEpy.bw_selection import improved_sheather_jones


from younger.commons.io import load_pickle, load_json, load_toml, save_json
from younger.commons.logging import logger

from younger.datasets.utils.constants import ONNXOperator, YoungerDatasetTask


# H:  Hierarchical; Q: Quantiles; K: KDE; R: Re-Clustering.
support_slc_modes = set([
    'HDBSCAN+H',
    'HDBSCAN+Q',
    'HDBSCAN+R',
    'KMeans+Q',
    'KMeans+R',
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


def find_all_hdbscan_subclusters(clusterer: hdbscan.HDBSCAN, cluster_labels: NDArray) -> dict[int, dict[int, list[int]]]:
    # Modified from here: https://github.com/scikit-learn-contrib/hdbscan/issues/401#issuecomment-2389803566
    tree_df = clusterer.condensed_tree_.to_pandas()
    
    def get_subclusters(node):
        children = tree_df[tree_df['parent'] == node]
        
        # If there are no children or only leaf children, return the node itself
        if children.empty or all(children['child'] < len(cluster_labels)):
            return {node: list(children[children['child'] < len(cluster_labels)]['child'])}
        
        # Recursively get subclusters for non-leaf children
        subclusters = {}
        for _, child in children.iterrows():
            child_id = int(child['child'])
            if child_id >= len(cluster_labels):
                subclusters.update(get_subclusters(child_id))
            else:
                subclusters[node] = subclusters.get(node, []) + [child_id]
        
        return subclusters
    
    def check_parent_recursive(points, node):
        children = tree_df[tree_df['parent'] == node]

        # Check if this node contains some points of label point
        child_points = children[children['child'].isin(points)]['child'].values
        if len(child_points) != 0:
            return True

        # Recursively search the parent for matching points
        for _, child in children.iterrows():
            child_id = int(child['child'])
            if child_id >= len(cluster_labels):
                result = check_parent_recursive(points, child_id)
                if result is not None:
                    return result

        return False
    
    all_subclusters = {}
    unique_labels = numpy.unique(cluster_labels)
    
    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_points = numpy.where(cluster_labels == label)[0]

            possible_candidates = tree_df[tree_df['child'].isin(cluster_points)]['parent'].unique()
            candidate_found = False
            cluster_nodes = []
            # The block of code that is most in use
            for candidate in possible_candidates:
                child_size = tree_df[tree_df['child'] == candidate]['child_size'].item()
                if child_size == len(cluster_points):
                    cluster_nodes.append(candidate)
                    candidate_found = True 
            # In theory, the block of code isn't most in use
            if not candidate_found:
                for candidate in possible_candidates:
                    if check_parent_recursive(cluster_points, candidate):
                        cluster_nodes.append(candidate)
                        candidate_found = True
            subclusters = {}
            for cluster_node in cluster_nodes:
                subclusters.update(get_subclusters(cluster_node))
            all_subclusters[label] = subclusters
    return all_subclusters


def kde_aic(bandwidth, ins_times):
    kde = sklearn.neighbors.KernelDensity(bandwidth=bandwidth)
    kde.fit(ins_times)
    log_likelihood = kde.score(ins_times)
    num_params = 2  # KDE has two parameters: bandwidth and kernel
    num_samples = ins_times.shape[0]
    return -2 * log_likelihood + 2 * num_params + (2 * num_params * (num_params + 1)) / (num_samples - num_params - 1)


def gmm_aic(n_components, ins_times):
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components)
    gmm.fit(ins_times)
    return gmm.aic(ins_times)


def fit(ins_times, fit_type='kde'):
    ins_times = numpy.array(ins_times).reshape(-1, 1)
    if fit_type == 'kde':
        # bandwidth_grid = [0.005, 0.01, 0.03, 0.07, 0.1]
        # best_bandwidth  = min(bandwidth_grid, key=lambda x: kde_aic(x, ins_times))
        best_bandwidth = improved_sheather_jones(ins_times)
        distribution_model = sklearn.neighbors.KernelDensity(bandwidth=best_bandwidth).fit(ins_times)
    if fit_type == 'gmm':
        n_components_grid = [2, 3, 4, 5, 6]
        best_n_components = min(n_components_grid, key=lambda x: gmm_aic(x, ins_times))
        distribution_model = sklearn.mixture.GaussianMixture(n_components=best_n_components).fit(ins_times)
    return distribution_model


def set_cover(objset: set[int], subsets: list[set[int]]) -> list[int]:
    max_subset_size = max([len(subset) for subset in subsets])
    size2ssids: list[set[int]] = [list() for i in range(1, max_subset_size+1)]
    element2ssids: dict[int, set[int]] = dict()

    for index, subset in enumerate(subsets):
        size2ssids[len(subset)].add(index)
        for element in subset:
            element2ssids[element] = element2ssids.get(element, set()).add(index)

    selected_ssids = set()
    selected_elements = set()

    while max_subset_size > 0:
        selected_ssid = size2ssids[max_subset_size].pop()
        selected_ssids.add(selected_ssid)
        selected_subset = subsets[selected_ssid]
        for element in selected_subset - selected_elements:
            for related_ssid in element2ssids[element]:
                related_subset = subsets[related_ssid]
                size2ssids[len(related_subset)].remove(related_ssid)
                size2ssids[len(related_subset)-1].add(related_ssid)
            selected_elements.add(element)
        if len(size2ssids[max_subset_size]) == 0:
            max_subset_size = max_subset_size - 1

    return selected_ssids


def select_graphs(
    dagopss: list[set[int]],
    dagembs: list[NDArray],
    opset: set[int],
    select_method: Literal['HDBSCAN+H', 'HDBSCAN+Q', 'HDBSCAN+R', 'KMeans+Q', 'KMeans+R'],
    select_config: dict[str, Any]
) -> tuple[list[int], list[int], list[int]]:
    cluster_method, segment_method = select_method.split('+')
    cluster_config = select_config.get(cluster_method, dict())

    select_any = select_config.get('select_any', False)
    assert isinstance(select_any, bool)
    select_num = select_config.get('select_num', 1)
    assert isinstance(select_num, int)
    select_rng = select_config.get('select_rng', 3) # For Q
    assert isinstance(select_rng, int)
    select_cfg = select_config.get('select_cfg', dict()) # For R
    assert isinstance(select_cfg, dict)
    select_ato = select_config.get('select_ato', False) # For R
    assert isinstance(select_ato, dict)

    select_noise_cfg = select_config.get('select_noise_cfg', dict())
    assert isinstance(select_noise_cfg, dict)

    noise_scores: dict[int, float] = dict()
    all_graph_scores: list[dict[int, float]] = list()
    if cluster_method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(**cluster_config).fit(numpy.array(dagembs))

        all_graph_scores: list[dict[int, float]] = [dict() for i in range(clusterer.labels_.max())]

        for index, (label, probability) in enumerate(zip(clusterer.labels_, clusterer.probabilities_)):
            if label == -1:
                noise_scores[index] = probability
            else:
                all_graph_scores[label][index] = probability

    if cluster_method == 'KMeans':
        clusterer = sklearn.cluster.KMeans(**cluster_config).fit(numpy.array(dagembs))

        all_graph_scores: list[dict[int, float]] = [dict() for i in range(clusterer.labels_.max())]

        euclidean = sklearn.metrics.DistanceMetric.get_metric('euclidean')
        for index, label in enumerate(clusterer.labels_):
            all_graph_scores[label][index] = - euclidean.pairwise(clusterer.cluster_centers_[label].reshape(1, -1), dagembs[index].reshape(1, -1))

    selected_noises = list()

    if len(noise_scores) != 0:
        select_noise_type = select_noise_cfg.get('type', 'any')
        assert select_noise_type in {'any', 'kms'}
        if select_noise_type == 'any':
            pass
        if select_noise_type == 'kms':
            select_noise_kmscfg = select_noise_cfg.get('kms_cfg', dict())
            select_noise_kmstop = select_noise_cfg.get('kms_top', dict())
            assert isinstance(select_noise_kmscfg, dict)
            noise_dagembs = [dagembs[index] for index in sorted(list(noise_scores.keys()))]
            noise_clusterer = sklearn.cluster.KMeans(**select_noise_kmscfg).fit(numpy.array(noise_dagembs))

            all_noise_scores: list[dict[int, float]] = [dict() for i in range(noise_clusterer.labels_.max())]

            euclidean = sklearn.metrics.DistanceMetric.get_metric('euclidean')
            for index, label in enumerate(noise_clusterer.labels_):
                all_noise_scores[label][index] = - euclidean.pairwise(noise_clusterer.cluster_centers_[label].reshape(1, -1), noise_dagembs[index].reshape(1, -1))

            for noise_scores in all_noise_scores:
                indices = [index for index in sorted(list(noise_scores.keys()), key=lambda index: noise_scores[index], reverse=True)]
                if select_any:
                    selected_noises.extend(numpy.random.choice(indices, size=select_noise_kmstop, replace=False).tolist())
                else:
                    selected_noises.extend(indices[:select_noise_kmstop])

    selected_graphs = list()

    if segment_method == 'H':
        all_subclusters = find_all_hdbscan_subclusters(clusterer, clusterer.labels_)
        for label, subclusters in all_subclusters.items():
            for subcluster_node, indices in subclusters.items():
                indices = sorted(indices, key=lambda index: all_graph_scores[label][index], reverse=True)
                if select_any:
                    selected_graphs.extend(numpy.random.choice(indices, size=select_num, replace=False).tolist())
                else:
                    selected_graphs.extend(indices[:select_num])

    if segment_method == 'Q':
        for label, graph_scores in enumerate(all_graph_scores):
            graph_scores = sorted(list(graph_scores.items()), key=lambda x: x[1], reverse=True)
            scores = [score for index, score in graph_scores]
            indexs = [index for index, score in graph_scores]

            bins: list[list[tuple[int, float]]] = [list() for _ in range(select_rng)]
            for i, bin_index in enumerate(numpy.digitize(scores, numpy.percentile(scores, numpy.linspace(0, 100, (select_rng)+1)), right=True)):
                bins[bin_index].append((indexs[i], scores[i]))

            for bin in bins:
                indices = [index for index, score in sorted(bin, key=lambda element: element[1], reverse=True)]
                if select_any:
                    selected_graphs.extend(numpy.random.choice(indices, size=select_num, replace=False).tolist())
                else:
                    selected_graphs.extend(indices[:select_num])

    if segment_method == 'R':
        for label, graph_scores in enumerate(all_graph_scores):
            in_cluster_dagembs = [dagembs[index] for index in list(graph_scores.keys())]

        if select_ato:
            possible_n_clusters = select_cfg.pop('n_clusters', 1)
            score_type = select_cfg.pop('score_type', 'dbi')
            assert score_type in {'dbi', 'chi'}

            best_n_clusters = possible_n_clusters
            sub_clusterer = sklearn.cluster.KMeans(n_clusters=possible_n_clusters, **select_cfg).fit(numpy.array(in_cluster_dagembs))
            if score_type == 'dbi':
                best_chi_score = sklearn.metrics.calinski_harabasz_score(numpy.array(in_cluster_dagembs), sub_clusterer.labels_)
            if score_type == 'chi':
                best_dbi_score = sklearn.metrics.davies_bouldin_score(numpy.array(in_cluster_dagembs), sub_clusterer.labels_)
            
            for n_clusters in range(1, possible_n_clusters):
                sub_clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, **select_cfg).fit(numpy.array(in_cluster_dagembs))
                if score_type == 'dbi':
                    chi_score = sklearn.metrics.calinski_harabasz_score(numpy.array(in_cluster_dagembs), sub_clusterer.labels_)
                    if best_chi_score < chi_score:
                        best_chi_score = chi_score
                        best_n_clusters = n_clusters
                if score_type == 'chi':
                    dbi_score = sklearn.metrics.davies_bouldin_score(numpy.array(in_cluster_dagembs), sub_clusterer.labels_)
                    if best_dbi_score > dbi_score:
                        best_dbi_score = chi_score
                        best_n_clusters = n_clusters
            select_cfg['n_clusters'] = best_n_clusters
        sub_clusterer = sklearn.cluster.KMeans(**select_cfg).fit(numpy.array(in_cluster_dagembs))

        sub_all_graph_scores: list[dict[int, float]] = [dict() for i in range(sub_clusterer.labels_.max())]

        euclidean = sklearn.metrics.DistanceMetric.get_metric('euclidean')
        for index, label in enumerate(sub_clusterer.labels_):
            sub_all_graph_scores[label][index] = - euclidean.pairwise(sub_clusterer.cluster_centers_[label].reshape(1, -1), in_cluster_dagembs[index].reshape(1, -1))

        for sub_graph_scores in sub_all_graph_scores:
            indices = [index for index in sorted(list(sub_graph_scores.keys()), key=lambda index: sub_graph_scores[index], reverse=True)]
            if select_any:
                selected_graphs.extend(numpy.random.choice(indices, size=select_num, replace=False).tolist())
            else:
                selected_graphs.extend(indices[:select_num])

    current_opset = set()
    all_selected = set(selected_graphs + selected_noises)
    for index in all_selected:
        current_opset.update(dagopss[index])
    uncovered_opset = opset - current_opset

    # print(uncovered_opset)
    subsets = list()
    subset_places = list()
    if len(uncovered_opset) != 0:
        for index, opset in enumerate(dagopss):
            if index not in all_selected:
                subsets.append(opset)
                subset_places.append(index)

    subset_indices = set_cover(uncovered_opset, subsets)
    supplied_graphs = list()
    for subset_index in subset_indices:
        supplied_graphs.append(subset_places[subset_index])
    return selected_graphs, selected_noises, supplied_graphs


def main(benchmark_dirpath: pathlib.Path, configuration_filepath: pathlib.Path):
    configuration = load_toml(configuration_filepath)

    system_requirements: dict = configuration['system_requirements']
    user_requirements: dict = configuration.get('user_requirements', dict())

    # [Begin] Load System Requirements and Bring the Real World
    providers = load_json(system_requirements['providers_filepath']) # Execution Provider Supportted Operator Sets
    default_provider_opset = set([str(op) for op in ONNXOperator.TYPES]) # Default CPU

    result = load_pickle(system_requirements['result_filepath'])
    real_world_opset = set([oplab for oplab, opemb in result['op_covered'].items()]) & default_provider_opset
    # [End] Load System Requirements and Bring the Real World

    # [Begin] Bring the World User Envision
    #  + [Begin] Setup Operator Set
    # Execution Provider
    user_provider: str = user_requirements.get('provider', None)
    # Platform
    user_platform: str = user_requirements.get('platform', None)
    # Operator Focus
    user_op_focus: str = user_requirements.get('op_focus', None)
    # Operator Sets:
    #     Set 1: The operator set specified by the user ('op_focus');
    #     Set 2: The operator set supported by the Execution Provider (EP) ('provider') on the platform ('platform') specified by the user;
    #     Set 3: The operator set supported by the ONNX CPU EP.
    # 
    # Priority:
    #     If only Set 1 is specified, use Set 1;
    #     If only Set 2 is specified, use Set 2;
    #     If both Sets 1 and 2 are specified, use the union of Sets 1 and 2.

    # v Set 1
    if user_op_focus is not None:
        specified_ur_opset: set[str] = set([str(tuple(op)) for op in user_op_focus])
    else:
        specified_ur_opset: set[str] = set()
    # ^ Set 1

    # v Set 2
    if user_provider is not None and user_platform is not None:
        # For more details about Execution Providers: https://onnxruntime.ai/docs/execution-providers/
        assert user_platform in support_platforms, f'Only Support Platform: GPU, CPU, and Edge! Yours Is {user_platform}'
        assert user_provider in support_providers, f'This Execution Provider - {user_provider} - Is Not Supportted Yet!'
        specified_ep_opset: set[str] = set([str(tuple(op)) for op in providers[user_provider][user_platform]]) # User Specified Execution CPU
    else:
        specified_ep_opset: set[str] = set()
    # ^ Set 2

    specified_opset = specified_ur_opset | specified_ep_opset

    unsupport_opset = specified_opset - real_world_opset
    specified_opset = specified_opset - unsupport_opset
    specified_opset = set([result['op_mapping']['o2i'][op] for op in specified_opset])
    logger.info(f'The following Operator user specified are not contained in Real World: {unsupport_opset}')
    #  - [End] Setup Operator Set

    #  + [Begin] Filter World User Envision
    # Task
    user_tk_focus: list = user_requirements.get('tk_focus', list())

    assert len(YoungerDatasetTask.I2T) == len(result['tk_mapping']) and all([df_tk == mt_tk for df_tk, mt_tk in zip()])
    specified_tasks = set(YoungerDatasetTask.I2T) | set(user_tk_focus)
    if len(specified_tasks) == 0:
        specified_tasks = set(YoungerDatasetTask.I2T)
        logger.info(f'The tasks user specified are not listed in Younger Supported Tasks. Now Use All Tasks.')
    else:
        logger.info(f'Tasks user specified are: {specified_tasks}.')
    specified_tasks = set([YoungerDatasetTask.T2I[tk] for tk in specified_tasks])

    # Loose Filter Opeartors
    user_op_loose: str = user_requirements.get('op_loose', False)

    real_world_dags2ps: list[tuple[str, str]] = list()
    real_world_dagembs: list[NDArray] = list()
    real_world_dagopss: list[set[int]] = list()
    for graph_hash, (dagops, dagemb, parent_graph_hash, graph_tasks) in result['dag_detail'].items():
        if (len(graph_tasks & specified_tasks) == 0) or (not user_op_loose and len(dagops - specified_opset) != 0):
            # Note:
            #     When the parameter 'op_loose' is set to true, operators supported by the CPU EP can be included in the generated world the user envisions.
            #     When the parameter 'op_loose' is set to false, select graphs with operators only specified by user.
            continue
        # (Cost, Operator Set, Cluster, Parent Graph Hash)
        real_world_dags2ps.append(graph_hash, parent_graph_hash)
        real_world_dagembs.append(numpy.ndarray(dagemb))
        real_world_dagopss.append(dagops)
    #  - [End] Filter World User Envision

    # # First Cover Operator/Cluster
    # user_priority: str = user_requirements.get('priority', 'OP')
    # assert user_priority in support_prioritys, f'Only Support Generation Cover Priority: OP (Operator) and CT (Cluster)! Yours Is {user_priority}'
    # First Clustering Method
    user_slc_mode: str = user_requirements.get('slc_mode', 'HDBSCAN+H')
    user_slc_args: str = user_requirements.get('slc_args', dict())
    assert user_slc_mode in support_slc_modes, f'Only Support Cluster Method: KMeans, HDBSCAN! Yours Is {user_slc_mode}'

    selected_graph_indices, selected_noise_indices = select_graphs(real_world_dagopss, real_world_dagembs, specified_opset, user_slc_mode, user_slc_args)

    benchmark = dict(
        selected_graph_hashes = [real_world_dags2ps[selected_graph_index] for selected_graph_index in selected_graph_indices],
        selected_noise_hashes = [real_world_dags2ps[selected_noise_index] for selected_noise_index in selected_noise_indices]
    )

    logger.info(f' v Saving Selected Graph Hashes of the Generated Benchmark ...')
    graph_hashes_of_generated_benchmark_filepath = benchmark_dirpath.joinpath('graph_hashes_of_generated_benchmark.json')
    save_json(benchmark, graph_hashes_of_generated_benchmark_filepath, indent=2)
    logger.info(f' ^ Done')