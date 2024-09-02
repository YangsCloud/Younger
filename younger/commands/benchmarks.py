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
        set_logger(YoungerHandle.BenchmarksName, mode='both', level='INFO', logging_filepath=logging_filepath)
        use_logger(YoungerHandle.BenchmarksName)


def run(arguments):
    pass


def others_phoronix_prepare_run(arguments):
    update_logger(arguments)
    download_xml = pathlib.Path(arguments.download_xml)
    download_dir = pathlib.Path(arguments.download_dir)

    from younger.benchmarks.others.phoronix import prepare_phoronix_onnx

    prepare_phoronix_onnx(download_xml, download_dir)


def others_phoronix_profile_run(arguments):
    update_logger(arguments)
    download_xml = pathlib.Path(arguments.download_xml)
    download_dir = pathlib.Path(arguments.download_dir)

    from younger.benchmarks.others.phoronix import profile_phoronix_onnx

    profile_phoronix_onnx(download_xml, download_dir)


def set_others_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    phoronix_prepare_parser = subparser.add_parser('phoronix-prepare')
    phoronix_prepare_parser.add_argument('--download-xml', type=str)
    phoronix_prepare_parser.add_argument('--download-dir', type=str)
    phoronix_prepare_parser.add_argument('--logging-filepath', type=str, default=None)
    phoronix_prepare_parser.set_defaults(run=others_phoronix_prepare_run)


def set_benchmarks_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    others_parser = subparser.add_parser('others')
    set_others_arguments(others_parser)

    parser.set_defaults(run=run)