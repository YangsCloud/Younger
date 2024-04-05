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

import sys
import argparse


from younger.commands.datasets import set_datasets_arguments
from younger.commands.benchmarks import set_benchmarks_arguments
from younger.commands.applications import set_applications_arguments


def get_command_line_argument_parser():
    argument_parser = argparse.ArgumentParser(allow_abbrev=True, formatter_class=argparse.RawTextHelpFormatter)
    arguments_subparser = argument_parser.add_subparsers()

    datasets_parser = arguments_subparser.add_parser('datasets')
    benchmarks_parser = arguments_subparser.add_parser('benchmarks')
    applications_parser = arguments_subparser.add_parser('applications')

    set_datasets_arguments(datasets_parser)
    set_datasets_arguments(benchmarks_parser)
    set_datasets_arguments(applications_parser)

    return argument_parser


def main():
    command_line_argument_parser = get_command_line_argument_parser()
    command_line_arguments = command_line_argument_parser.parse_args()

    print(
        f'                >   Welcome to use Younger!   <                 \n'
        f'----------------------------------------------------------------\n'
        f'Please use the following command to make the most of the system:\n'
        f'0. younger --help\n'
        f'1. younger datasets --help\n'
        f'2. younger benchmarks --help\n'
        f'3. younger applications --help\n'
    )
    sys.stdout.flush()


if __name__ == '__main__':
    main()