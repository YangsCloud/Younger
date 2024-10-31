#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 12:15
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import argparse

from younger.commons.logging import set_logger, use_logger
from younger.commons.constants import YoungerHandle


def update_logger(arguments):
    if arguments.logging_filepath:
        logging_filepath = pathlib.Path(arguments.logging_filepath)
        set_logger(YoungerHandle.DatasetsName, mode='both', level='INFO', logging_filepath=logging_filepath)
        use_logger(YoungerHandle.DatasetsName)


def run(arguments):
    pass


def convert_run(arguments):
    pass


def retrieve_run(arguments):
    pass


def filter_run(arguments):
    update_logger(arguments)
    dataset_dirpath = pathlib.Path(arguments.dataset_dirpath)
    save_dirpath = pathlib.Path(arguments.save_dirpath)

    from younger.datasets.constructors.official import filter

    filter.main(
        dataset_dirpath, save_dirpath,
        arguments.worker_number,
        arguments.max_inclusive_version,
        arguments.clean,
    )


def split_run(arguments):
    update_logger(arguments)
    tasks_filepath = pathlib.Path(arguments.tasks_filepath) if arguments.tasks_filepath is not None else None
    dataset_dirpath = pathlib.Path(arguments.dataset_dirpath) if arguments.dataset_dirpath is not None else None
    save_dirpath = pathlib.Path(arguments.save_dirpath) if arguments.save_dirpath is not None else None

    if arguments.mode == 'ego':
        from younger.datasets.constructors.official import ego_split
        ego_split.main(
            dataset_dirpath, save_dirpath,
            range_left=arguments.range_left,
            range_right=arguments.range_right,
            sample_frequency=arguments.sample_frequency,
            worker_number=arguments.worker_number,
            seed=arguments.seed,
        )
    elif arguments.mode == 'community':
        from younger.datasets.constructors.official import community_split
        community_split.main(
            tasks_filepath, dataset_dirpath, save_dirpath,
            arguments.version,
            arguments.silly,
            arguments.node_size_lbound, arguments.node_size_ubound,
            arguments.edge_size_lbound, arguments.edge_size_ubound,
            arguments.train_proportion, arguments.valid_proportion, arguments.test_proportion,
            arguments.partition_number,
            arguments.worker_number,
            arguments.seed
        )
    elif arguments.mode == 'random':
        from younger.datasets.constructors.official import random_split
        random_split.main(
            dataset_dirpath, save_dirpath,
            arguments.version,
            arguments.allow_domains,
            arguments.subgraph_sizes,
            arguments.subgraph_number,
            arguments.retrieve_try,
            arguments.node_size_lbound, arguments.node_size_ubound,
            arguments.edge_size_lbound, arguments.edge_size_ubound,
            arguments.train_proportion, arguments.valid_proportion, arguments.test_proportion,
            arguments.seed
        )
    else:
        from younger.datasets.constructors.official import split
        split.main(
            tasks_filepath, dataset_dirpath, save_dirpath,
            arguments.version,
            arguments.silly,
            arguments.metric_name,
            arguments.node_size_lbound, arguments.node_size_ubound,
            arguments.edge_size_lbound, arguments.edge_size_ubound,
            arguments.train_proportion, arguments.valid_proportion, arguments.test_proportion,
            arguments.partition_number,
            arguments.worker_number,
            arguments.seed
        )


def statistics_run(arguments):
    update_logger(arguments)
    dataset_dirpath = pathlib.Path(arguments.dataset_dirpath)
    save_dirpath = pathlib.Path(arguments.save_dirpath)

    from younger.datasets.constructors.official import statistics

    statistics.main(dataset_dirpath, save_dirpath, arguments.tasks, arguments.dataset_names, arguments.dataset_splits, arguments.metric_names, arguments.worker_number)


def convert_huggingface_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)
    model_ids_filepath = pathlib.Path(arguments.model_ids_filepath)
    status_filepath = pathlib.Path(arguments.status_filepath)

    from younger.datasets.constructors.huggingface import convert

    convert.main(save_dirpath, cache_dirpath, model_ids_filepath, status_filepath, device=arguments.device, model_size_threshold=arguments.model_size_threshold, huggingface_token=arguments.huggingface_token, mode=arguments.mode)


def retrieve_huggingface_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)

    from younger.datasets.constructors.huggingface import retrieve

    kwargs = dict(
        library=arguments.library,
        label=arguments.label,
        token=arguments.token,
        worker_number=arguments.worker_number,
        force_reload=arguments.force_reload,
    )

    retrieve.main(arguments.mode, save_dirpath, cache_dirpath, arguments.min_json, **kwargs)


def convert_onnx_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)
    model_ids_filepath = pathlib.Path(arguments.model_ids_filepath)
    status_filepath = pathlib.Path(arguments.status_filepath)

    from younger.datasets.constructors.onnx import convert

    convert.main(save_dirpath, cache_dirpath, model_ids_filepath, status_filepath)


def convert_torchvision_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)
    model_ids_filepath = pathlib.Path(arguments.model_ids_filepath)
    status_filepath = pathlib.Path(arguments.status_filepath)

    from younger.datasets.constructors.torchvision import convert

    convert.main(save_dirpath, cache_dirpath, model_ids_filepath, status_filepath)


def retrieve_onnx_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)

    from younger.datasets.constructors.onnx import retrieve

    kwargs = dict(
        force_reload=arguments.force_reload,
    )

    retrieve.main(arguments.mode, save_dirpath, cache_dirpath, arguments.min_json, **kwargs)


def retrieve_torchvision_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)

    from younger.datasets.constructors.torchvision import retrieve

    kwargs = dict(
        force_reload=arguments.force_reload,
    )

    retrieve.main(arguments.mode, save_dirpath, cache_dirpath, arguments.min_json, **kwargs)


def api_run(arguments):
    update_logger(arguments)
    dataset_dirpath = pathlib.Path(arguments.dataset_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)
    memory_dirpath = pathlib.Path(arguments.memory_dirpath)

    if arguments.type == 'handle_complete':
        from younger.datasets.constructors.official.api import handle_complete

        handle_complete.main(
            dataset_dirpath, cache_dirpath, memory_dirpath,
            arguments.worker_number,
            arguments.since_version,
            arguments.paper,
            arguments.token,
            arguments.proxy,
        )

    if arguments.type == 'handle_filter':
        from younger.datasets.constructors.official.api import handle_filter

        handle_filter.main(
            dataset_dirpath, cache_dirpath, memory_dirpath,
            arguments.worker_number,
            arguments.meta,
            arguments.with_attributes,
            arguments.since_version,
            arguments.paper,
            arguments.token,
            arguments.proxy,
        )


def set_datasets_convert_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    huggingface_parser = subparser.add_parser('huggingface')
    huggingface_parser.add_argument('--model-ids-filepath', type=str, required=True)
    huggingface_parser.add_argument('--save-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--cache-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--status-filepath', type=str, default='./status.flg')
    huggingface_parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    huggingface_parser.add_argument('--model-size-threshold', type=int, default=3*1024*1024*1024)
    huggingface_parser.add_argument('--logging-filepath', type=str, default=None)
    huggingface_parser.add_argument('--huggingface-token', type=str, default=None)
    huggingface_parser.add_argument('--mode', type=str, choices=['optimum', 'onnx', 'keras', 'tflite'], default='optimum')
    huggingface_parser.set_defaults(run=convert_huggingface_run)

    onnx_parser = subparser.add_parser('onnx')
    onnx_parser.add_argument('--model-ids-filepath', type=str, required=True)
    onnx_parser.add_argument('--save-dirpath', type=str, default='.')
    onnx_parser.add_argument('--cache-dirpath', type=str, default='.')
    onnx_parser.add_argument('--status-filepath', type=str, default='./status.flg')
    onnx_parser.add_argument('--logging-filepath', type=str, default=None)
    onnx_parser.set_defaults(run=convert_onnx_run)

    torchvision_parser = subparser.add_parser('torchvision')
    torchvision_parser.add_argument('--model-ids-filepath', type=str, required=True)
    torchvision_parser.add_argument('--save-dirpath', type=str, default='.')
    torchvision_parser.add_argument('--cache-dirpath', type=str, default='.')
    torchvision_parser.add_argument('--status-filepath', type=str, default='./status.flg')
    torchvision_parser.add_argument('--logging-filepath', type=str, default=None)
    torchvision_parser.set_defaults(run=convert_torchvision_run)

    parser.set_defaults(run=convert_run)


def set_datasets_retrieve_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    huggingface_parser = subparser.add_parser('huggingface')
    huggingface_parser.add_argument('--mode', type=str, choices=['Models', 'Model_Infos', 'Model_IDs', 'Metrics', 'Tasks'], required=True)
    huggingface_parser.add_argument('--save-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--cache-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--library', type=str, default=None)
    huggingface_parser.add_argument('--token', type=str, default=None)
    huggingface_parser.add_argument('--worker-number', type=int, default=10)
    huggingface_parser.add_argument('--label', action='store_true')
    huggingface_parser.add_argument('--min-json', action='store_true')
    huggingface_parser.add_argument('--force-reload', action='store_true')
    huggingface_parser.add_argument('--logging-filepath', type=str, default=None)
    huggingface_parser.set_defaults(run=retrieve_huggingface_run)

    onnx_parser = subparser.add_parser('onnx')
    onnx_parser.add_argument('--mode', type=str, choices=['Models', 'Model_Infos', 'Model_IDs'], required=True)
    onnx_parser.add_argument('--save-dirpath', type=str, default='.')
    onnx_parser.add_argument('--cache-dirpath', type=str, default='.')
    onnx_parser.add_argument('--min-json', action='store_true')
    onnx_parser.add_argument('--force-reload', action='store_true')
    onnx_parser.add_argument('--logging-filepath', type=str, default=None)
    onnx_parser.set_defaults(run=retrieve_onnx_run)

    torchvision_parser = subparser.add_parser('torchvision')
    torchvision_parser.add_argument('--mode', type=str, choices=['Models', 'Model_Infos', 'Model_IDs'], required=True)
    torchvision_parser.add_argument('--save-dirpath', type=str, default='.')
    torchvision_parser.add_argument('--cache-dirpath', type=str, default='.')
    torchvision_parser.add_argument('--min-json', action='store_true')
    torchvision_parser.add_argument('--force-reload', action='store_true')
    torchvision_parser.add_argument('--logging-filepath', type=str, default=None)
    torchvision_parser.set_defaults(run=retrieve_torchvision_run)

    parser.set_defaults(run=retrieve_run)


def set_datasets_filter_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dataset-dirpath', type=str, required=True)
    parser.add_argument('--save-dirpath', type=str, default='.')

    parser.add_argument('--max-inclusive-version', type=int, default=None)

    parser.add_argument('--clean', action='store_true')

    parser.add_argument('--worker-number', type=int, default=4)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=filter_run)


def set_datasets_split_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--tasks-filepath', type=str, default=None)
    parser.add_argument('--dataset-dirpath', type=str, default=None)
    parser.add_argument('--save-dirpath', type=str, default=None)

    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--silly', action='store_true')
    parser.add_argument('--allow-domains', type=str, nargs='*', default=[])
    parser.add_argument('--subgraph-sizes', type=int, nargs='+', default=[5])
    parser.add_argument('--subgraph-number', type=int, default=10)
    parser.add_argument('--retrieve-try', type=int, default=1000)
    parser.add_argument('--mode', type=str, choices=['ego', 'community', 'random'], default='community')
    parser.add_argument('--metric-name', type=str, default=None)

    parser.add_argument('--sample-frequency', type=int, default=None)
    parser.add_argument('--range-left', type=int, default=None)
    parser.add_argument('--range-right', type=int, default=None)

    parser.add_argument('--node-size-lbound', type=int, default=None)
    parser.add_argument('--node-size-ubound', type=int, default=None)
    parser.add_argument('--edge-size-lbound', type=int, default=None)
    parser.add_argument('--edge-size-ubound', type=int, default=None)

    parser.add_argument('--train-proportion', type=float, default=80)
    parser.add_argument('--valid-proportion', type=float, default=10)
    parser.add_argument('--test-proportion', type=float, default=10)

    parser.add_argument('--partition-number', type=int, default=10)

    parser.add_argument('--worker-number', type=int, default=4)

    parser.add_argument('--seed', type=int, default=16861)

    parser.add_argument('--logging-filepath', type=str, default=None)

    parser.set_defaults(run=split_run)


def set_datasets_statistics_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dataset-dirpath', type=str, required=True)
    parser.add_argument('--save-dirpath', type=str, default='.')

    parser.add_argument('--tasks', type=str, nargs='*', default=[])
    parser.add_argument('--dataset-names', type=str, nargs='*', default=[])
    parser.add_argument('--dataset-splits', type=str, nargs='*', default=[])
    parser.add_argument('--metric-names', type=str, nargs='*', default=[])

    parser.add_argument('--worker-number', type=int, default=4)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=statistics_run)


def set_datasets_api_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--dataset-dirpath', type=str, required=True)
    parser.add_argument('--cache-dirpath', type=str, default='.')
    parser.add_argument('--memory-dirpath', type=str, default='./memory')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--with-attributes', action='store_true')
    parser.add_argument('--since-version', type=str, default='0.0.0')
    parser.add_argument('--paper', action='store_true')
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--proxy', type=str, default=None)

    parser.add_argument('--worker-number', type=int, default=4)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=api_run)


def set_datasets_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    convert_parser = subparser.add_parser('convert')
    retrieve_parser = subparser.add_parser('retrieve')
    filter_parser = subparser.add_parser('filter')
    split_parser = subparser.add_parser('split')
    statistics_parser = subparser.add_parser('statistics')
    api_parser = subparser.add_parser('api')

    set_datasets_convert_arguments(convert_parser)
    set_datasets_retrieve_arguments(retrieve_parser)
    set_datasets_filter_arguments(filter_parser)
    set_datasets_split_arguments(split_parser)
    set_datasets_statistics_arguments(statistics_parser)
    set_datasets_api_arguments(api_parser)

    parser.set_defaults(run=run)
