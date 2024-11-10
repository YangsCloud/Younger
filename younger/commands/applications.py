#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 12:16
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import argparse


def run(arguments):
    pass


def deep_learning_run(arguments):
    pass


def deep_learning_train_run(arguments):
    task_name = arguments.task_name
    config_filepath = pathlib.Path(arguments.config_filepath)
    checkpoint_dirpath = pathlib.Path(arguments.checkpoint_dirpath)

    from younger.applications.deep_learning import train

    train(
        task_name,
        config_filepath,

        checkpoint_dirpath,
        arguments.checkpoint_name,
        arguments.keep_number,

        arguments.train_batch_size,
        arguments.valid_batch_size,
        arguments.shuffle,

        arguments.checkpoint_filepath,
        arguments.reset_optimizer,
        arguments.reset_period,

        arguments.life_cycle,
        arguments.report_period,
        arguments.update_period,
        arguments.train_period,
        arguments.valid_period,

        arguments.device,
        arguments.world_size,
        arguments.master_addr,
        arguments.master_port,
        arguments.master_rank,

        arguments.seed,
        arguments.make_deterministic,
    )


def deep_learning_test_run(arguments):
    task_name = arguments.task_name
    config_filepath = pathlib.Path(arguments.config_filepath)
    checkpoint_filepath = pathlib.Path(arguments.checkpoint_filepath)

    from younger.applications.deep_learning import test

    test(
        task_name,
        config_filepath,
        checkpoint_filepath,
        arguments.test_batch_size,
        arguments.device,
    )


def deep_learning_cli_run(arguments):
    task_name = arguments.task_name
    config_filepath = pathlib.Path(arguments.config_filepath)
    checkpoint_filepath = pathlib.Path(arguments.checkpoint_filepath)

    from younger.applications.deep_learning import cli

    cli(
        task_name,
        config_filepath,
        checkpoint_filepath,
        arguments.device,
    )


def deep_learning_api_run(arguments):
    task_name = arguments.task_name
    config_filepath = pathlib.Path(arguments.config_filepath)
    checkpoint_filepath = pathlib.Path(arguments.checkpoint_filepath)

    from younger.applications.deep_learning import api

    api(
        task_name,
        config_filepath,
        checkpoint_filepath,
        arguments.device,
    )


def set_applications_deep_learning_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train')
    train_parser.add_argument('--task-name', type=str)
    train_parser.add_argument('--config-filepath', type=str)

    train_parser.add_argument('--checkpoint-dirpath', type=str)
    train_parser.add_argument('--checkpoint-name', type=str, default='checkpoint')
    train_parser.add_argument('--keep-number', type=int, default=50)

    train_parser.add_argument('--train-batch-size', type=int, default=32)
    train_parser.add_argument('--valid-batch-size', type=int, default=32)
    train_parser.add_argument('--shuffle', action='store_true')

    train_parser.add_argument('--checkpoint-filepath', type=str, default=None)
    train_parser.add_argument('--reset-optimizer', action='store_true')
    train_parser.add_argument('--reset-period', action='store_true')

    train_parser.add_argument('--life-cycle', type=int, default=100)
    train_parser.add_argument('--report-period', type=int, default=100)
    train_parser.add_argument('--update-period', type=int, default=1)
    train_parser.add_argument('--train-period', type=int, default=1000)
    train_parser.add_argument('--valid-period', type=int, default=1000)

    train_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')

    train_parser.add_argument('--world-size', type=int, default=1)
    train_parser.add_argument('--master-addr', type=str, default='localhost')
    train_parser.add_argument('--master-port', type=str, default='16161')
    train_parser.add_argument('--master-rank', type=int, default=0)

    train_parser.add_argument('--seed', type=int, default=1234)
    train_parser.add_argument('--make-deterministic', action='store_true')
    train_parser.set_defaults(run=deep_learning_train_run)

    test_parser = subparser.add_parser('test')
    test_parser.add_argument('--task-name', type=str)
    test_parser.add_argument('--config-filepath', type=str)

    test_parser.add_argument('--checkpoint-filepath', type=str)
    test_parser.add_argument('--test-batch-size', type=int, default=32)

    test_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
    test_parser.set_defaults(run=deep_learning_test_run)

    api_parser = subparser.add_parser('cli')
    api_parser.add_argument('--task-name', type=str)
    api_parser.add_argument('--config-filepath', type=str)
    api_parser.add_argument('--checkpoint-filepath', type=str)
    api_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
    api_parser.set_defaults(run=deep_learning_cli_run)


    api_parser = subparser.add_parser('api')
    api_parser.add_argument('--task-name', type=str)
    api_parser.add_argument('--config-filepath', type=str)
    api_parser.add_argument('--checkpoint-filepath', type=str)
    api_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
    api_parser.set_defaults(run=deep_learning_api_run)

    parser.set_defaults(run=deep_learning_run)


def set_applications_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    deep_learning_parser = subparser.add_parser('deep_learning')

    set_applications_deep_learning_arguments(deep_learning_parser)

    parser.set_defaults(run=run)
