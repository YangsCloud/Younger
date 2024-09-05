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

from typing import Literal

from younger.commons.logging import set_logger, use_logger
from younger.commons.constants import YoungerHandle


def update_logger(arguments):
    if arguments.logging_filepath:
        logging_filepath = pathlib.Path(arguments.logging_filepath)
        set_logger(YoungerHandle.BenchmarksName, mode='both', level='INFO', logging_filepath=logging_filepath)
        use_logger(YoungerHandle.BenchmarksName)


def run(arguments):
    pass


def prepare_run(arguments):
    update_logger(arguments)
    bench_dirpath = pathlib.Path(arguments.bench_dirpath)
    dataset_dirpath = pathlib.Path(arguments.dataset_dirpath)

    from younger.benchmarks.prepare import main

    main(bench_dirpath, dataset_dirpath, arguments.version)


def analyze_run(arguments):
    update_logger(arguments)
    dataset_dirpath = pathlib.Path(arguments.dataset_dirpath)
    statistics_dirpath = pathlib.Path(arguments.statistics_dirpath)

    from younger.benchmarks.analyze import main

    main(dataset_dirpath, statistics_dirpath)


def produce_run(arguments):
    update_logger(arguments)
    real_dirpath = pathlib.Path(arguments.real_dirpath)
    product_dirpath = pathlib.Path(arguments.product_dirpath)
    statistics_dirpath = pathlib.Path(arguments.statistics_dirpath) if arguments.statistics_dirpath is not None else None

    from younger.benchmarks.produce import main

    main(real_dirpath, product_dirpath, statistics_dirpath)


def set_prepare_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--bench-dirpath', type=str)
    parser.add_argument('--dataset-dirpath', type=str)
    parser.add_argument('--version', type=str)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=prepare_run)


def set_analyze_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dataset-dirpath', type=str)
    parser.add_argument('--statistics-dirpath', type=str)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=analyze_run)


def set_produce_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--real-dirpath', type=str)
    parser.add_argument('--product-dirpath', type=str)
    parser.add_argument('--statistics-dirpath', type=str, default=None)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=produce_run)


def set_benchmarks_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    prepare_parser = subparser.add_parser('prepare')
    set_prepare_arguments(prepare_parser)

    analyze_parser = subparser.add_parser('analyze')
    set_analyze_arguments(analyze_parser)

    produce_parser = subparser.add_parser('produce')
    set_produce_arguments(produce_parser)

    parser.set_defaults(run=run)