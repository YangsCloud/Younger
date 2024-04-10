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


def convert_huggingface_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)
    model_ids_filepath = pathlib.Path(arguments.model_ids_filepath)

    from younger.datasets.constructors.huggingface import convert

    convert.main(save_dirpath, cache_dirpath, model_ids_filepath, device=arguments.device, threshold=arguments.threshold)


def retrieve_huggingface_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)

    from younger.datasets.constructors.huggingface import retrieve

    retrieve.main(arguments.mode, save_dirpath, cache_dirpath, library=arguments.library)


def convert_onnx_run(arguments):
    update_logger(arguments)
    save_dirpath = pathlib.Path(arguments.save_dirpath)
    cache_dirpath = pathlib.Path(arguments.cache_dirpath)

    from younger.datasets.constructors.onnx import convert

    convert.main(save_dirpath, cache_dirpath)


def set_datasets_convert_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    huggingface_parser = subparser.add_parser('huggingface')
    huggingface_parser.add_argument('--model-ids-filepath', type=str, required=True)
    huggingface_parser.add_argument('--save-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--cache-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    huggingface_parser.add_argument('--threshold', type=int, default=3*1024*1024*1024)
    huggingface_parser.add_argument('--logging-filepath', type=str, default=None)
    huggingface_parser.set_defaults(run=convert_huggingface_run)

    onnx_parser = subparser.add_parser('onnx')
    onnx_parser.add_argument('--save-dirpath', type=str, default='.')
    onnx_parser.add_argument('--cache-dirpath', type=str, default='.')
    onnx_parser.add_argument('--logging-filepath', type=str, default=None)
    onnx_parser.set_defaults(run=convert_onnx_run)

    parser.set_defaults(run=convert_run)


def set_datasets_retrieve_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    huggingface_parser = subparser.add_parser('huggingface')
    huggingface_parser.add_argument('--mode', type=str, choices=['Models', 'Model_Infos', 'Model_IDs', 'Metrics'], required=True)
    huggingface_parser.add_argument('--save-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--cache-dirpath', type=str, default='.')
    huggingface_parser.add_argument('--library', type=str, default='transformers')
    huggingface_parser.add_argument('--logging-filepath', type=str, default=None)
    huggingface_parser.set_defaults(run=retrieve_huggingface_run)

    parser.set_defaults(run=retrieve_run)


def set_datasets_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    convert_parser = subparser.add_parser('convert')
    retrieve_parser = subparser.add_parser('retrieve')

    set_datasets_convert_arguments(convert_parser)
    set_datasets_retrieve_arguments(retrieve_parser)

    parser.set_defaults(run=run)