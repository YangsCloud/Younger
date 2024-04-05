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

from younger.datasets.constructors.huggingface import convert as hf_convert
from younger.datasets.constructors.onnx import convert as ox_convert


def convert(args):
    save_dirpath = pathlib.Path(args.save_dirpath)
    cache_dirpath = pathlib.Path(args.cache_dirpath)

    if args.hub == 'huggingface':
        hf_convert.main(save_dirpath, cache_dirpath, args.library, device=args.device, threshold=args.threshold)

    if args.hub == 'onnx':
        ox_convert.main(save_dirpath, cache_dirpath)


def set_datasets_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()
    convert_parser = subparser.add_parser('convert')
    convert_parser.add_argument('--hub', type=str, choices=['huggingface', 'onnx'], help='Choose Hub')
    convert_parser.add_argument('--save-dirpath', type=str, default='.')
    convert_parser.add_argument('--cache-dirpath', type=str, default='.')
    convert_parser.add_argument('--library', type=str, default='transformers')
    convert_parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    convert_parser.add_argument('--threshold', type=int, default=3*1024*1024)
    convert_parser.set_defaults(func=convert)