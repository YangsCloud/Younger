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


import ast
import umap
import hdbscan
import numpy
import pathlib
import xlsxwriter

import sklearn.cluster
import sklearn.manifold
import matplotlib.pyplot

from typing import Literal
from datetime import datetime

from younger.commons.io import load_json, save_json, load_pickle, save_pickle, load_toml
from younger.commons.logging import logger

from younger.datasets.modules import Dataset, Network
from younger.datasets.utils.translation import get_operator_origin


class HDBSCAN(hdbscan.HDBSCAN):
    def predict(self, points_to_predict):
        labels, _ = hdbscan.prediction.approximate_predict(self, points_to_predict)
        return labels


reducer_initializers: dict[str, sklearn.base.BaseEstimator] = {
    't-SNE': sklearn.manifold.TSNE,
    'UMAP': umap.UMAP,
}


cluster_initializers: dict[str, sklearn.base.BaseEstimator] = {
    'KMeans': sklearn.cluster.KMeans,
    'AffinityPropagation': sklearn.cluster.AffinityPropagation,
    'MeanShift': sklearn.cluster.MeanShift,
    'HDBSCAN': HDBSCAN,
}


def statistically_analyze(dataset_name: str, dataset_dirpath: pathlib.Path, sts_results_dirpath: pathlib.Path) -> dict[str, int | dict[str, tuple[int, float]]]:
    logger.info(f' v Now statistically analyzing {dataset_name} ...')

    sts_results = dict()

    total_ops = 0
    op_type_frequency = dict()
    op_origin_frequency = dict()
    unknown_op_type_frequency = dict()
    for instance in Dataset.load_instances(dataset_dirpath):
        try:
            graph = Network.standardize(instance.network.graph)
        except:
            # Already cleansed.
            graph = instance.network.graph
        if graph.number_of_nodes() == 0:
            continue
        total_ops += graph.number_of_nodes()
        for node_index in graph.nodes():
            op_type = Network.get_node_identifier_from_features(graph.nodes[node_index], mode='type')
            op_origin = get_operator_origin(graph.nodes[node_index]['operator']['op_type'], graph.nodes[node_index]['operator']['domain'])
            if op_origin != 'unknown':
                op_type_frequency[op_type] = op_type_frequency.get(op_type, 0) + 1
                op_origin_frequency[op_origin] = op_origin_frequency.get(op_origin, 0) + 1
            else:
                unknown_op_type_frequency[op_type] = unknown_op_type_frequency.get(op_type, 0) + 1
    sts_results['op_type_frequency'] = op_type_frequency
    sts_results['op_origin_frequency'] = op_origin_frequency
    sts_results['unknown_op_type_frequency'] = unknown_op_type_frequency
    logger.info(f'   Total operators = {total_ops}')
    logger.info(f'   Total different operator types = {len(op_type_frequency)}')
    logger.info(f'   Total different operator origins = {len(op_origin_frequency)}')

    sts_results['total_ops'] = total_ops
    for op_type, frequency in sts_results['op_type_frequency'].items():
        sts_results['op_type_frequency'][op_type] = (frequency, frequency/total_ops)

    for op_origin, frequency in sts_results['op_origin_frequency'].items():
        sts_results['op_origin_frequency'][op_origin] = (frequency, frequency/total_ops)

    for unknown_op_type, frequency in sts_results['unknown_op_type_frequency'].items():
        sts_results['unknown_op_type_frequency'][unknown_op_type] = (frequency, frequency/total_ops)

    # v =================================== Save To File =================================== v
    # Save Statistical Analysis Results (JSON)
    json_filepath = sts_results_dirpath.joinpath(f'sts_results_{dataset_name}.json')
    save_json(sts_results, json_filepath, indent=2)
    logger.info(f'   {dataset_name.capitalize()}\'s statistical analysis results (JSON format) saved into: {json_filepath}')

    # Save Statistical Analysis Results (XLSX)
    xlsx_filepath = sts_results_dirpath.joinpath(f'sts_results_{dataset_name}.xlsx')
    workbook = xlsxwriter.Workbook(xlsx_filepath)

    # op type frequency
    worksheet = workbook.add_worksheet('op_type_frequency')

    worksheet.write(0, 0, 'OP_Name')
    worksheet.write(0, 1, 'OP_Domain')
    worksheet.write(0, 2, 'Frequency')
    worksheet.write(0, 3, 'Ratio')

    for index, (op_type, (frequency, ratio)) in enumerate(sts_results['op_type_frequency'].items(), start=1):
        op_name, op_domain = ast.literal_eval(op_type)
        worksheet.write(index, 0, op_name)
        worksheet.write(index, 1, op_domain)
        worksheet.write(index, 2, frequency)
        worksheet.write(index, 3, ratio)

    # op origin frequency
    worksheet = workbook.add_worksheet('op_origin_frequency')

    worksheet.write(0, 0, 'OP_Origin')
    worksheet.write(0, 1, 'Frequency')
    worksheet.write(0, 2, 'Ratio')

    for index, (op_origin, (frequency, ratio)) in enumerate(sts_results['op_origin_frequency'].items(), start=1):
        worksheet.write(index, 0, op_origin)
        worksheet.write(index, 1, frequency)
        worksheet.write(index, 2, ratio)

    workbook.close()
    logger.info(f'   {dataset_name.capitalize()}\'s statistical analysis results (XLSX format) saved into: {xlsx_filepath}')
    # ^ =================================== Save To File =================================== ^

    logger.info(f' ^ Done')
    return sts_results


def statistical_analysis(sts_results_dirpath: pathlib.Path, configuration_filepath: pathlib.Path):
    configuration = load_toml(configuration_filepath)
    younger_dataset = configuration['sts'].get('younger_dataset', None)
    compare_datasets = configuration['sts'].get('compare_datasets', list())
    assert younger_dataset is not None

    younger_dataset_sts_results = statistically_analyze(younger_dataset['name'], pathlib.Path(younger_dataset['path']), sts_results_dirpath)

    if len(compare_datasets) != 0:
        compare_dataset_sts_results = dict()
        for compare_dataset in compare_datasets:
            compare_dataset_sts_results[compare_dataset['name']] = statistically_analyze(compare_dataset['name'], pathlib.Path(compare_dataset['path']), sts_results_dirpath)

        if len(compare_dataset_sts_results) != 0:
            logger.info(f' v Analyzing Younger Compare To Other Datasets ...')
            for dataset_name, dataset_sts_results in compare_dataset_sts_results.items():
                op_type_cover_ratios = list() # Other Cover Younger
                uncovered_op_types = list() # Other Uncovered By Younger
                for op_type, (frequency, ratio) in dataset_sts_results['op_type_frequency'].items():
                    if op_type in younger_dataset_sts_results['op_type_frequency']:
                        op_type_cover_ratios.append((op_type, frequency / younger_dataset_sts_results['op_type_frequency'][op_type][0]))
                    else:
                        uncovered_op_types.append(op_type)

                op_origin_cover_ratios = list() # Other Cover Younger
                uncovered_op_origins = list() # Other Uncovered By Younger
                for op_origin, (frequency, ratio) in dataset_sts_results['op_origin_frequency'].items():
                    if op_origin in younger_dataset_sts_results['op_origin_frequency']:
                        op_origin_cover_ratios.append((op_origin, frequency / younger_dataset_sts_results['op_origin_frequency'][op_origin][0]))
                    else:
                        uncovered_op_origins.append(op_origin)

                compare_sts_results = dict(
                    op_type_cover_ratios = op_type_cover_ratios,
                    uncovered_op_types = uncovered_op_types,
                    op_origin_cover_ratios = op_origin_cover_ratios,
                    uncovered_op_origins = uncovered_op_origins
                )

                json_filepath = sts_results_dirpath.joinpath(f'sts_results_compare_{dataset_name}.json')
                save_json(compare_sts_results, json_filepath, indent=2)
                logger.info(f'   {dataset_name.capitalize()}\'s statistical analysis results (JSON format) compared to Younger saved into: {json_filepath}')

            logger.info(f' ^ Done')
        #figure_filepath = stc_results_dirpath.joinpath(f'stc_visualization_sketch.pdf')


def structural_analysis(stc_results_dirpath: pathlib.Path, configuration_filepath: pathlib.Path):
    configuration = load_toml(configuration_filepath)
    younger_result = configuration['stc'].get('younger_result', None)
    assert younger_result is not None
    younger_result_dict = load_pickle(younger_result['path'])

    compare_results = configuration['stc'].get('compare_results', list())
    compare_result_dicts = dict()
    for compare_result in compare_results:
        compare_result_dicts[compare_result['name']] = load_pickle(compare_result['path'])

    timestamp = datetime.now()
    timestamp_string = timestamp.strftime("%Y%m%d_%H%M%S")

    younger_oplabs = [oplab for oplab, opemb in sorted(list(younger_result_dict['op_covered'].items()))]
    younger_opembs = [opemb for oplab, opemb in sorted(list(younger_result_dict['op_covered'].items()))]

    younger_daglabs = [daglab for daglab, (opset, dagemb, parent_graph_hash) in sorted(list(younger_result_dict['dag_detail'].items()))]
    younger_dagembs = [dagemb for daglab, (opset, dagemb, parent_graph_hash) in sorted(list(younger_result_dict['dag_detail'].items()))]

    op_cluster_type = configuration['stc'].get('op_cluster_type', 'HDBSCAN')
    op_reducer_type = configuration['stc'].get('op_reducer_type', 'UMAP')
    op_cluster_kwargs = configuration['stc'].get('op_cluster_kwargs', dict())
    op_reducer_kwargs = configuration['stc'].get('op_reducer_kwargs', dict())

    dag_cluster_type = configuration['stc'].get('dag_cluster_type', 'HDBSCAN')
    dag_reducer_type = configuration['stc'].get('dag_reducer_type', 'UMAP')
    dag_cluster_kwargs = configuration['stc'].get('dag_cluster_kwargs', dict())
    dag_reducer_kwargs = configuration['stc'].get('dag_reducer_kwargs', dict())

    logger.info(f'   + Training Cluster for Younger OP Embeddings ({op_cluster_type}).')
    op_cluster_initializer = cluster_initializers.get(op_cluster_type, hdbscan.HDBSCAN)
    opembs_cluster = op_cluster_initializer(**op_cluster_kwargs).fit(numpy.array(younger_opembs))
    logger.info(f'   - Done.')

    logger.info(f'   + Training Cluster for Younger DAG Embeddings ({dag_cluster_type}).')
    dag_cluster_initializer = cluster_initializers.get(dag_cluster_type, hdbscan.HDBSCAN)
    dagembs_cluster = dag_cluster_initializer(**dag_cluster_kwargs).fit(numpy.array(younger_dagembs))
    logger.info(f'   - Done.')

    logger.info(f'   + Fitting Reducer for Younger OP Embeddings ({op_reducer_type}).')
    op_reducer_initializer = reducer_initializers.get(op_reducer_type, umap.UMAP)
    opembs_reducer = op_reducer_initializer(**op_reducer_kwargs).fit(numpy.array(younger_opembs))
    logger.info(f'   - Done.')

    logger.info(f'   + Fitting Reducer for Younger DAG Embeddings ({dag_reducer_type}).')
    dag_reducer_initializer = reducer_initializers.get(dag_reducer_type, umap.UMAP)
    dagembs_reducer = dag_reducer_initializer(**dag_reducer_kwargs).fit(numpy.array(younger_dagembs))
    logger.info(f'   - Done.')

    younger_reduced_opembs = opembs_reducer.transform(numpy.array(younger_opembs))
    younger_reduced_dagembs = dagembs_reducer.transform(numpy.array(younger_dagembs))

    logger.info(f'   Total {numpy.unique(opembs_cluster.labels_)} OP Clusters.')
    logger.info(f'   Total {numpy.unique(dagembs_cluster.labels_)} DAG Clusters.')

    # v Plot Sketch Figure (Younger Part)
    fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(20, 10))
    axes[0].scatter(younger_reduced_opembs[:, 0],  younger_reduced_opembs[:, 1],  c=opembs_cluster.labels_, cmap='Paired', marker='.', s=10**2, zorder=1, alpha=1)
    axes[0].set_title('Operators')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')
    axes[0].legend()

    axes[1].scatter(younger_reduced_dagembs[:, 0], younger_reduced_dagembs[:, 1], c=dagembs_cluster.labels_, cmap='Paired', marker='.', s=10**2, zorder=1, alpha=1)
    axes[1].set_title('Graphs')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')
    axes[1].legend()
    figure_filepath = stc_results_dirpath.joinpath(f'stc_visualization_sketch_{timestamp_string}.pdf')
    matplotlib.pyplot.tight_layout()
    fig.savefig(figure_filepath)
    logger.info(f'   Structural analysis results are visualized, and the figure is saved into: {figure_filepath}')
    # ^ Plot Sketch Figure (Younger Part)

    if len(compare_result_dicts) != 0:
        younger_op_indices = {oplab: index for index, (oplab, opemb) in enumerate(sorted(list(younger_result_dict['op'].items())))}
        # v Plot Sketch Figure (Compare Part)
        for compare_result_name, compare_result_dict in compare_result_dicts.items():
            compare_oplabs = [oplab for oplab, opemb in compare_result_dict['op'].items()]
            compare_opembs = [opemb for oplab, opemb in compare_result_dict['op'].items()]

            compare_daglabs = [daglab for daglab, dagemb in compare_result_dict['dag'].items()]
            compare_dagembs = [dagemb for daglab, dagemb in compare_result_dict['dag'].items()]

            compare_reduced_opembs = numpy.array([younger_reduced_opembs[younger_op_indices[oplab]] for oplab in compare_oplabs])
            compare_opembs_labels = opembs_cluster.predict(numpy.array(compare_opembs))

            compare_reduced_dagembs = dagembs_reducer.transform(numpy.array(compare_dagembs))
            compare_dagembs_labels = dagembs_cluster.predict(numpy.array(compare_dagembs))

            fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(20, 10))
            sc_1 = axes[0].scatter(
                younger_reduced_opembs[:, 0],  younger_reduced_opembs[:, 1],
                c=opembs_cluster.labels_, cmap='Paired', marker='.', s=10**2, zorder=1, alpha=1,
            )
            axes[0].scatter(
                compare_reduced_opembs[:, 0],  compare_reduced_opembs[:, 1],
                c=compare_opembs_labels, cmap='Paired', edgecolor='red', linewidth=1, marker='*', s=12**2, zorder=2,# alpha=0.6,
            )

            sc_2 = axes[1].scatter(
                younger_reduced_dagembs[:, 0], younger_reduced_dagembs[:, 1],
                c=dagembs_cluster.labels_, cmap='Paired', marker='.', s=10**2, zorder=1, alpha=1,
            )
            axes[1].scatter(
                compare_reduced_dagembs[:, 0], compare_reduced_dagembs[:, 1],
                c=compare_dagembs_labels, cmap='Paired', edgecolor='red', linewidth=1, marker='*', s=12**2, zorder=2,# alpha=0.6,
            )

            axes[0].set_title('Operators')
            axes[0].set_xlabel('X-axis')
            axes[0].set_ylabel('Y-axis')
            axes[0].legend(*sc_1.legend_elements(), title=f'OP: Younger v.s. {compare_result_name}')

            axes[1].set_title('Graphs')
            axes[1].set_xlabel('X-axis')
            axes[1].set_ylabel('Y-axis')
            axes[1].legend(*sc_2.legend_elements(), title=f'DAG: Younger v.s. {compare_result_name}')
            figure_filepath = stc_results_dirpath.joinpath(f'stc_visualization_sketch_compare_{compare_result_name}_{timestamp_string}.pdf')
            matplotlib.pyplot.tight_layout()
            fig.savefig(figure_filepath)
            logger.info(f'   Structural analysis results are visualized, and the figure is saved into: {figure_filepath}')
        # ^ Plot Sketch Figure (Compare Part)


def main(results_dirpath: pathlib.Path, configuration_filepath: pathlib.Path, mode: Literal['sts', 'stc', 'both'] = 'sts'):
    assert mode in {'sts', 'stc', 'both'}
    analyzed = False
    if mode in {'sts', 'both'}:
        statistical_analysis(results_dirpath.joinpath('statistical'), configuration_filepath)
        analyzed = True

    if mode in {'stc', 'both'}:
        structural_analysis(results_dirpath.joinpath('structural'), configuration_filepath)
        analyzed = True

    if analyzed:
        logger.info(f' = Analyzed Younger and Other Datasets.')