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

import argparse


from younger.commons.logging import naive_log
from younger.commands.datasets import set_datasets_arguments
from younger.commands.benchmarks import set_benchmarks_arguments
from younger.commands.applications import set_applications_arguments


def run(arguments):
    naive_log(
        f'                                                                \n'
        f'                >   Welcome to use Younger!   <                 \n'
        f'----------------------------------------------------------------\n'
        f'Please use the following command to make the most of the system:\n'
        f'0. younger --help                                               \n'
        f'1. younger datasets --help                                      \n'
        f'2. younger benchmarks --help                                    \n'
        f'3. younger applications --help                                  \n'
        f'                                                                \n'
    )


def main():
    argument_parser = argparse.ArgumentParser(allow_abbrev=True, formatter_class=argparse.RawTextHelpFormatter)

    arguments_subparser = argument_parser.add_subparsers()

    datasets_parser = arguments_subparser.add_parser('datasets')
    benchmarks_parser = arguments_subparser.add_parser('benchmarks')
    applications_parser = arguments_subparser.add_parser('applications')

    set_datasets_arguments(datasets_parser)
    set_benchmarks_arguments(benchmarks_parser)
    set_applications_arguments(applications_parser)

    argument_parser.set_defaults(run=run)

    arguments = argument_parser.parse_args()

    #try:
    arguments.run(arguments)
    #except Exception as exception:
    #    argument_parser.print_help()


if __name__ == '__main__':
    main()
