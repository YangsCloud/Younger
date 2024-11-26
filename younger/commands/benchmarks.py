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

    main(bench_dirpath, dataset_dirpath, arguments.version, arguments.direct)


def analyze_run(arguments):
    update_logger(arguments)
    results_dirpath = pathlib.Path(arguments.results_dirpath)
    configuration_filepath = pathlib.Path(arguments.configuration_filepath)

    from younger.benchmarks.analyze import main

    main(results_dirpath, configuration_filepath, mode=arguments.mode)


def generate_run(arguments):
    update_logger(arguments)
    benchmark_dirpath = pathlib.Path(arguments.benchmark_dirpath)
    configuration_filepath = pathlib.Path(arguments.configuration_filepath)

    from younger.benchmarks.generate import main

    main(benchmark_dirpath, configuration_filepath)


def merge_run(arguments):
    update_logger(arguments)
    real_dirpath = pathlib.Path(arguments.real_dirpath)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    other_dirpaths = [pathlib.Path(other_dirpath) for other_dirpath in arguments.other_dirpaths]

    from younger.benchmarks.merge import main

    main(real_dirpath, save_dirpath, other_dirpaths)


def set_prepare_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--bench-dirpath', type=str)
    parser.add_argument('--dataset-dirpath', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--direct', type=str, choices=['instance', 'onnx', 'both', None], default=None)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=prepare_run)


def set_analyze_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('-r', '--results-dirpath', type=str)
    parser.add_argument('-c', '--configuration-filepath', type=str)
    parser.add_argument('-m', '--mode', type=str, choices=['sts', 'stc', 'both'])

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=analyze_run)


def set_generate_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('-b', '--benchmark-dirpath', type=str)
    parser.add_argument('-c', '--configuration-filepath', type=str)

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=generate_run)


def set_merge_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--real-dirpath', type=str)
    parser.add_argument('--save-dirpath', type=str)
    parser.add_argument('--other-dirpaths', type=str, nargs='+')

    parser.add_argument('--logging-filepath', type=str, default=None)
    parser.set_defaults(run=merge_run)


def set_benchmarks_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    prepare_parser = subparser.add_parser('prepare')
    set_prepare_arguments(prepare_parser)

    analyze_parser = subparser.add_parser('analyze')
    set_analyze_arguments(analyze_parser)

    generate_parser = subparser.add_parser('generate')
    set_generate_arguments(generate_parser)

    merge_parser = subparser.add_parser('merge')
    set_merge_arguments(merge_parser)

    parser.set_defaults(run=run)