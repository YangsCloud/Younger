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

from younger.commons.logging import set_logger, use_logger
from younger.commons.constants import YoungerHandle


def update_logger(arguments):
    if arguments.logging_filepath:
        logging_filepath = pathlib.Path(arguments.logging_filepath)
        set_logger(YoungerHandle.DatasetsName, mode='both', level='INFO', logging_filepath=logging_filepath)
        use_logger(YoungerHandle.DatasetsName)


def run(arguments):
    pass


def performance_prediction_run(arguments):
    pass


def performance_prediction_train_run(arguments):
    update_logger(arguments)
    train_dataset_dirpath = pathlib.Path(arguments.train_dataset_dirpath)
    valid_dataset_dirpath = pathlib.Path(arguments.valid_dataset_dirpath)
    checkpoint_dirpath = pathlib.Path(arguments.checkpoint_dirpath)

    from younger.applications.performance_prediction.run import train

    train(
        train_dataset_dirpath,
        valid_dataset_dirpath,
        checkpoint_dirpath,
        arguments.mode,
        arguments.x_feature_get_type,
        arguments.y_feature_get_type,

        arguments.node_dim,
        arguments.task_dim,
        arguments.dataset_dim,
        arguments.hidden_dim,
        arguments.readout_dim,
        arguments.cluster_num,

        arguments.checkpoint_filepath,
        arguments.checkpoint_name,
        arguments.keep_number,
        arguments.reset_optimizer,
        arguments.reset_period,
        arguments.fine_tune,

        arguments.life_cycle,
        arguments.update_period,
        arguments.train_period,
        arguments.valid_period,
        arguments.report_period,

        arguments.train_batch_size,
        arguments.valid_batch_size,
        arguments.learning_rate,
        arguments.weight_decay,
        arguments.shuffle,

        arguments.device,
        arguments.world_size,
        arguments.master_addr,
        arguments.master_port,
        arguments.master_rank,
        arguments.worker_number,

        arguments.seed,
        arguments.make_deterministic,
    )


def performance_prediction_test_run(arguments):
    update_logger(arguments)
    test_dataset_dirpath = pathlib.Path(arguments.test_dataset_dirpath)
    checkpoint_filepath = pathlib.Path(arguments.checkpoint_filepath)

    from younger.applications.performance_prediction.run import test

    test(
        test_dataset_dirpath,
        checkpoint_filepath,
        arguments.x_feature_get_type,
        arguments.y_feature_get_type,

        arguments.test_batch_size,

        arguments.node_dim,
        arguments.task_dim,
        arguments.dataset_dim,
        arguments.hidden_dim,
        arguments.readout_dim,
        arguments.cluster_num,

        arguments.device,
        arguments.worker_number,
    )


def set_applications_performance_perdiction_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train')
    train_parser.add_argument('--train-dataset-dirpath', type=str)
    train_parser.add_argument('--valid-dataset-dirpath', type=str)
    train_parser.add_argument('--checkpoint-dirpath', type=str)
    train_parser.add_argument('--mode', type=str, choices=['Supervised', 'Unsupervised'], default='Unsupervised')
    train_parser.add_argument('--x-feature-get-type', type=str, choices=['OnlyOp'], default='OnlyOp')
    train_parser.add_argument('--y-feature-get-type', type=str, choices=['OnlyMt', 'TkDsMt'], default='OnlyMt')

    train_parser.add_argument('--node-dim', type=int, default=512)
    train_parser.add_argument('--task-dim', type=int, default=512)
    train_parser.add_argument('--dataset-dim', type=int, default=512)
    train_parser.add_argument('--hidden-dim', type=int, default=512)
    train_parser.add_argument('--readout-dim', type=int, default=256)
    train_parser.add_argument('--cluster-num', type=int, default=16)

    train_parser.add_argument('--checkpoint-filepath', type=str, default=None)
    train_parser.add_argument('--checkpoint-name', type=str, default='checkpoint')
    train_parser.add_argument('--keep-number', type=int, default=50)
    train_parser.add_argument('--reset-optimizer', action='store_true')
    train_parser.add_argument('--reset-period', action='store_true')
    train_parser.add_argument('--fine-tune', action='store_true')

    train_parser.add_argument('--life-cycle', type=int, default=100)
    train_parser.add_argument('--update-period', type=int, default=1)
    train_parser.add_argument('--train-period', type=int, default=1000)
    train_parser.add_argument('--valid-period', type=int, default=1000)
    train_parser.add_argument('--report-period', type=int, default=100)

    train_parser.add_argument('--train-batch-size', type=int, default=32)
    train_parser.add_argument('--valid-batch-size', type=int, default=32)
    train_parser.add_argument('--learning-rate', type=float, default=1e-3)
    train_parser.add_argument('--weight-decay', type=float, default=1e-1)
    train_parser.add_argument('--shuffle', action='store_true')

    train_parser.add_argument('--world-size', type=int, default=1)
    train_parser.add_argument('--master-addr', type=str, default='localhost')
    train_parser.add_argument('--master-port', type=str, default='16161')
    train_parser.add_argument('--master-rank', type=int, default=0)
    train_parser.add_argument('--worker-number', type=int, default=4)

    train_parser.add_argument('--seed', type=int, default=1234)

    train_parser.add_argument('--make-deterministic', action='store_true')

    train_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')

    train_parser.add_argument('--logging-filepath', type=str, default=None)
    train_parser.set_defaults(run=performance_prediction_train_run)

    test_parser = subparser.add_parser('test')
    test_parser.add_argument('--test-dataset-dirpath', type=str)
    test_parser.add_argument('--checkpoint-filepath', type=str)
    test_parser.add_argument('--x-feature-get-type', type=str, choices=['OnlyOp'], default='OnlyOp')
    test_parser.add_argument('--y-feature-get-type', type=str, choices=['OnlyMt', 'TkDsMt'], default='OnlyMt')

    test_parser.add_argument('--test-batch-size', type=int, default=32)

    test_parser.add_argument('--node-dim', type=int, default=512)
    test_parser.add_argument('--task-dim', type=int, default=512)
    test_parser.add_argument('--dataset-dim', type=int, default=512)
    test_parser.add_argument('--hidden-dim', type=int, default=512)
    test_parser.add_argument('--readout-dim', type=int, default=256)
    test_parser.add_argument('--cluster-num', type=int, default=16)

    test_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
    test_parser.add_argument('--worker-number', type=int, default=4)

    test_parser.add_argument('--logging-filepath', type=str, default=None)
    test_parser.set_defaults(run=performance_prediction_test_run)

    parser.set_defaults(run=performance_prediction_run)


def set_applications_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    performance_prediction_parser = subparser.add_parser('performance_prediction')

    set_applications_performance_perdiction_arguments(performance_prediction_parser)

    parser.set_defaults(run=run)
