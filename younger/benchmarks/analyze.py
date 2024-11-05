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
import numpy
import pathlib
import xlsxwriter

import matplotlib.pyplot

from typing import Any, Literal
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from younger.commons.io import load_json, save_json, load_pickle, save_pickle, load_toml
from younger.commons.logging import logger

from younger.datasets.modules import Dataset, Network
from younger.datasets.utils.translation import get_operator_origin

from younger.applications.utils.neural_network import load_operator_embedding


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


def statistical_analysis(younger_dataset_dirpath: pathlib.Path, sts_results_dirpath: pathlib.Path, other_dataset_indices_filepath: pathlib.Path | None = None):
    younger_dataset_sts_results = statistically_analyze('younger', younger_dataset_dirpath, sts_results_dirpath)

    if other_dataset_indices_filepath is not None:
        other_dataset_sts_results = dict()
        with open(other_dataset_indices_filepath, 'r') as f:
            for line in f:
                other_dataset_name, other_dataset_dirpath = line.split(':')[0].strip(), line.split(':')[1].strip()
                other_dataset_sts_results[other_dataset_name] = statistically_analyze(other_dataset_name, other_dataset_dirpath, sts_results_dirpath)

        if len(other_dataset_sts_results) != 0:
            logger.info(f' v Analyzing Younger Compare To Other Datasets ...')
            for dataset_name, dataset_sts_results in other_dataset_sts_results.items():
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


def structurally_analyze(dataset_name: str, dataset_dirpath: pathlib.Path, stc_results_dirpath: pathlib.Path, operator_embedding_dict: dict[str, NDArray[numpy.float64]]) -> dict[str, dict[str, list[float]]]:
    logger.info(f' v Now structurally analyzing {dataset_name} ...')

    stc_results = dict()

    dag_embs = dict()
    op_types = set()
    for instance in Dataset.load_instances(dataset_dirpath):
        try:
            graph = Network.standardize(instance.network.graph)
        except:
            # Already cleansed.
            graph = instance.network.graph
        if graph.number_of_nodes() == 0:
            continue
        dag_emb = 0
        for node_index in graph.nodes():
            op_type = Network.get_node_identifier_from_features(graph.nodes[node_index], mode='type')
            op_type = op_type if op_type in operator_embedding_dict else '__UNK__'
            op_types.add(op_type)
            dag_emb += operator_embedding_dict[op_type]
        dag_emb = dag_emb / graph.number_of_nodes()

        dag_embs[instance.labels['model_name'][0]] = dag_emb.tolist()

    op_embs = dict()
    for op_type in op_types:
        op_embs[op_type] = operator_embedding_dict[op_type].tolist()
    stc_results['op_embeddings'] = op_embs
    stc_results['dag_embeddings'] = dag_embs

    return stc_results


def structural_analysis(younger_dataset_dirpath: pathlib.Path, stc_results_dirpath: pathlib.Path, other_dataset_indices_filepath: pathlib.Path | None = None, operator_embedding_dirpath: pathlib.Path | None = None, visualize_configuration_filepath: pathlib.Path | None = None, standardization: bool = False):
    visualize_configuration = load_toml(visualize_configuration_filepath)
    opemb_weights, opemb_op_dict = load_operator_embedding(operator_embedding_dirpath)
    operator_embedding_dict = dict()
    for operator_id, index in opemb_op_dict.items():
        operator_embedding_dict[operator_id] = opemb_weights[index]

    younger_dataset_stc_results = structurally_analyze('younger', younger_dataset_dirpath, stc_results_dirpath, operator_embedding_dict)
    younger_oplabs = numpy.array([oplab for oplab, opemb in younger_dataset_stc_results['op_embeddings'].items()])
    younger_opembs = numpy.array([opemb for oplab, opemb in younger_dataset_stc_results['op_embeddings'].items()])
    opembs_reducer = umap.UMAP(**visualize_configuration['OP'])
    logger.info(f'   + Fitting Operator Embeddings UMAP Reducer.')
    opembs_reducer.fit(numpy.array(younger_opembs))
    logger.info(f'   - Done.')
    younger_opembs_reduced = opembs_reducer.transform(numpy.array(younger_opembs))

    younger_daglabs = numpy.array([daglab for daglab, dagemb in younger_dataset_stc_results['dag_embeddings'].items()])
    younger_dagembs = numpy.array([dagemb for daglab, dagemb in younger_dataset_stc_results['dag_embeddings'].items()])
    dagembs_reducer = umap.UMAP(**visualize_configuration['DAG'])
    logger.info(f'   + Fitting DAG Embeddings UMAP Reducer.')
    dagembs_reducer.fit(younger_dagembs)
    logger.info(f'   - Done.')
    younger_dagembs_reduced = dagembs_reducer.transform(younger_dagembs)

    younger_stc_results = dict(
        oplabs = younger_oplabs,
        opembs = younger_opembs.tolist(),
        opmebs_reduced = younger_opembs_reduced.tolist(),
        opembs_reducer_config = visualize_configuration['OP'],
        daglabs = younger_daglabs,
        dagembs = younger_dagembs.tolist(),
        dagmebs_reduced = younger_dagembs_reduced.tolist(),
        dagembs_reducer_config = visualize_configuration['DAG'],
    )

    pickle_filepath = stc_results_dirpath.joinpath(f'stc_results_younger.pkl')
    save_pickle(younger_stc_results, pickle_filepath)
    logger.info(f'   Younger\'s structural analysis results (JSON format) compared to Younger saved into: {pickle_filepath}')

    # v Plot Sketch Figure (Younger Part)
    colormap = matplotlib.pyplot.get_cmap('tab20')
    younger_color_op = colormap(0)
    younger_color_dag = colormap(1)
    fig, axes = matplotlib.pyplot.subplots(2, 2, figsize=(20, 20))
    axes[0].scatter(younger_opembs_reduced[:, 0],  younger_opembs_reduced[:, 1],  color=younger_color_op,  label='younger')
    axes[0].set_title('Operators')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')
    axes[0].legend()

    axes[1].scatter(younger_dagembs_reduced[:, 0], younger_dagembs_reduced[:, 1], color=younger_color_dag, label='younger')
    axes[1].set_title('Graphs')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')
    axes[1].legend()
    # ^ Plot Sketch Figure (Younger Part)

    # v Plot Sketch Figure (Compare Part)
    fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(20, 8))
    axes[2].scatter(younger_opembs_reduced[:, 0],  younger_opembs_reduced[:, 1],  color=younger_color_op,  label='younger')
    axes[3].scatter(younger_dagembs_reduced[:, 0], younger_dagembs_reduced[:, 1], color=younger_color_dag, label='younger')
    # - Plot Sketch Figure (Compare Part)

    if other_dataset_indices_filepath is not None:
        other_dataset_stc_results = dict()
        with open(other_dataset_indices_filepath, 'r') as f:
            for line in f:
                other_dataset_name, other_dataset_dirpath = line.split(':')[0].strip(), line.split(':')[1].strip()
                other_dataset_stc_results[other_dataset_name] = structurally_analyze(other_dataset_name, other_dataset_dirpath, stc_results_dirpath, operator_embedding_dict)

        if len(other_dataset_stc_results) != 0:
            for index, (dataset_name, dataset_stc_results) in enumerate(other_dataset_stc_results.items()):
                compare_color_op = colormap(0 + 2 * (index + 1))
                compare_color_dag = colormap(1 + 2 * (index + 1))

                compare_oplabs = numpy.array([oplab for oplab, opemb in dataset_stc_results['op_embeddings'].items()])
                compare_opembs = numpy.array([opemb for oplab, opemb in dataset_stc_results['op_embeddings'].items()])
                compare_opembs_reduced = opembs_reducer.transform(compare_opembs)

                compare_daglabs = numpy.array([daglab for daglab, dagemb in dataset_stc_results['dag_embeddings'].items()])
                compare_dagembs = numpy.array([dagemb for daglab, dagemb in dataset_stc_results['dag_embeddings'].items()])
                compare_dagembs_reduced = opembs_reducer.transform(compare_dagembs)
                compare_stc_results = dict(
                    oplabs = compare_oplabs,
                    opembs = compare_opembs.tolist(),
                    opmebs_reduced = compare_opembs_reduced.tolist(),
                    daglabs = compare_daglabs,
                    dagembs = compare_dagembs.tolist(),
                    dagmebs_reduced = compare_dagembs_reduced.tolist(),
                )

                pickle_filepath = stc_results_dirpath.joinpath(f'stc_results_compare_{dataset_name}.json')
                save_pickle(compare_stc_results, pickle_filepath)
                logger.info(f'   {dataset_name.capitalize()}\'s structural analysis results compared to Younger saved into: {pickle_filepath}')

                # - Plot Sketch Figure (Compare Part)
                axes[2].scatter(compare_opembs_reduced[:, 0],  compare_opembs_reduced[:, 1],  color=compare_color_op,  label=dataset_name, marker='*')
                axes[3].scatter(compare_dagembs_reduced[:, 0], compare_dagembs_reduced[:, 1], color=compare_color_dag, label=dataset_name, marker='*')
                # - Plot Sketch Figure (Compare Part)

    axes[2].set_title('Operators')
    axes[2].set_xlabel('X-axis')
    axes[2].set_ylabel('Y-axis')
    axes[2].legend()

    axes[3].set_title('Graphs')
    axes[3].set_xlabel('X-axis')
    axes[3].set_ylabel('Y-axis')
    axes[3].legend()
    figure_filepath = stc_results_dirpath.joinpath(f'stc_visualization_sketch.pdf')
    matplotlib.pyplot.tight_layout()
    fig.savefig(figure_filepath)
    logger.info(f'   Structural analysis results are visualized, and the figure is saved into: {figure_filepath}')
    # ^ Plot Sketch Figure (Compare Part)


def main(younger_dataset_dirpath: pathlib.Path, results_dirpath: pathlib.Path, other_dataset_indices_filepath: pathlib.Path | None = None, operator_embedding_dirpath: pathlib.Path | None = None, visualize_configuration_filepath: pathlib.Path | None = None, standardization: bool = False, mode: Literal['sts', 'stc', 'both'] = 'sts'):
    assert mode in {'sts', 'stc', 'both'}
    analyzed = False
    if mode in {'sts', 'both'}:
        statistical_analysis(younger_dataset_dirpath, results_dirpath.joinpath('statistical'), other_dataset_indices_filepath)
        analyzed = True

    if mode in {'stc', 'both'}:
        structural_analysis(younger_dataset_dirpath, results_dirpath.joinpath('structural'), other_dataset_indices_filepath, operator_embedding_dirpath, visualize_configuration_filepath, standardization)
        analyzed = True

    if analyzed:
        logger.info(f' = Analyzed Younger and Other Datasets.')