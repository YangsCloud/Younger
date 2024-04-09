#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-06 18:27
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import torch
import numpy
import pathlib

from torch import distributed
from typing import Literal

from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from younger.commons.logging import logger

from younger.applications.utils.neural_network import get_model_parameters_number, get_device_descriptor, fix_random_procedure, load_checkpoint, save_checkpoint
from younger.applications.performance_prediction.models import NAPPGNNBase
from younger.applications.performance_prediction.datasets import YoungerDataset


def infer_cluster_num(dataset: Dataset) -> int:
    pass


def get_logging_metrics_str(metric_names: list[str], metric_values: list[str]) -> str:
    metrics_str = str()
    for metric_name, metric_value in zip(metric_names, metric_values):
        metrics_str += f'{metric_name}: {metric_value}'
    return metrics_str


def period_operation(
    is_master: bool, world_size: int,
    epoch: int, step: int, train_period: int, valid_period: int, report_period: int, start_position: int,
    record_unit: Literal['Epoch', 'Step'], current_unit: Literal['Epoch', 'Step'],
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dirpath: str, checkpoint_name: str, keep_number: int,
    valid_dataloader: DataLoader,
    losses: torch.Tensor, loss_names: list[str],
    device_descriptor: torch.device,
    is_distribution
):
    if current_unit != record_unit:
        return

    if current_unit == 'Epoch':
        position = epoch
    if current_unit == 'Step':
        position = step
    
    if position % train_period == 0 and is_master:
        logger.info('  -> Saving checkpoint ...')
        tic = time.time()
        checkpoint = dict()
        checkpoint[record_unit] = position + start_position
        checkpoint['model_state'] = model.state_dict()
        checkpoint['optimizer_state'] = optimizer.state_dict()
        save_checkpoint(checkpoint, checkpoint_path=checkpoint_dirpath, checkpoint_name=checkpoint_name, record_unit=record_unit, keep_number=keep_number)
        toc = time.time()
        logger.info(f'  -> Checkpoint is saved to \'{checkpoint_dirpath}\' at {position} {record_unit} (Take: {toc-tic:2.0f}s)')        

    if position % valid_period == 0:
        distributed.barrier()
        if is_master:
            exact_check(device_descriptor, model, valid_dataloader, 'Valid')
        distributed.barrier()

    if position % report_period == 0:
        if is_distribution:
            distributed.all_reduce(losses, op = distributed.ReduceOp.SUM)
            losses = losses / world_size
        metric_names = ['Total-Loss', 'Regression-Loss (MSE)', 'Cluster-loss']
        metric_values = [f'{float(loss):.4f}' for loss in losses]
        logger.info(f'  {record_unit}@{position} - {get_logging_metrics_str(metric_names, metric_values)}')


def exact_check(device_descriptor: torch.device, model: torch.nn.Module, dataloader: DataLoader, split: Literal['Valid', 'Test']):
    model.eval()
    logger.info(f'  v {split} Begin ...')
    overall_loss = 0
    overall_main_loss = 0
    overall_gnn_pooling_loss = 0
    metric_names = ['Total-Loss', 'Regression-Loss (MSE)', 'Cluster-loss']
    tic = time.time()
    with torch.no_grad():
        for index, data in enumerate(dataloader, start=1):
            data: Data = data.to(device_descriptor)
            output, gnn_pooling_loss = model(data.x, data.edge_index, data.y[:, 0], data.batch)
            main_loss = torch.nn.functional.mse_loss(output, data.y[:, 1])
            loss = main_loss + gnn_pooling_loss
            metric_values = [f'{float(loss):.4f}', f'{float(main_loss):.4f}', f'{float(gnn_pooling_loss):.4f}']
            logger.info(f'  Process No.{index} - {get_logging_metrics_str(metric_names, metric_values)}')
            overall_loss += loss
            overall_main_loss += main_loss
            overall_gnn_pooling_loss += gnn_pooling_loss
    toc = time.time()
    metric_values = [f'{float(overall_loss/index):.4f}', f'{float(overall_main_loss/index):.4f}', f'{float(overall_gnn_pooling_loss/index):.4f}']
    logger.info(f'  ^  {split} Finished. Overall Result - {get_logging_metrics_str(metric_names, metric_values)} (Time Cost = {toc-tic:.2f}s)')


def exact_train(
    rank: int,
    model: torch.nn.Module, train_dataset: Dataset, valid_dataset: Dataset,
    checkpoint_filepath: str, checkpoint_dirpath: str, checkpoint_name: str, keep_number: int, reset_optimizer: bool, reset_period: bool, fine_tune: bool,
    life_cycle:int, train_period: int, valid_period: int, report_period: int, record_unit: Literal['Epoch', 'Step'],
    train_batch_size: int, valid_batch_size: int, learning_rate: float, weight_decay: float,
    seed: int, device: Literal['CPU', 'GPU'], world_size: int, master_rank: int, is_distribution: bool,
):
    is_master = rank == master_rank
    fix_random_procedure(seed)
    device_descriptor = get_device_descriptor(device, rank)

    if is_master:
        logger.disabled = False
    else:
        logger.disabled = True

    model.to(device_descriptor)
    logger.info(f'  Model Moved to Device \'{device_descriptor}\'')

    if is_distribution:
        distributed.init_process_group('nccl', rank=rank, world_size=world_size)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    if is_distribution:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, sampler=train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    if rank == master_rank:
        valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    checkpoint = load_checkpoint(pathlib.Path(checkpoint_filepath), checkpoint_name, record_unit=record_unit)

    if checkpoint is None:
        logger.info(f'  Train from scratch.')
    else:
        logger.info(f'  Train from checkpoint [\'{checkpoint_filepath}\'] at [{checkpoint[record_unit]}] {record_unit}.')

        if reset_optimizer:
            logger.info(f'  Reset Optimizer.')
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        logger.info(f'  v Loading Parameters ...')
        model.load_state_dict(checkpoint['model_state'])
        logger.info(f'  ^ Loaded.')

        if reset_period:
            logger.info(f'  Reset {record_unit}.')
            start_position = 0
        else:
            start_position = checkpoint[record_unit]

    logger.info(f'Training Start ...')
    logger.info(f'  Train Life Cycle: Total {life_cycle} Epochs!')
    logger.info(f'  Saving checkpoint every {train_period} {record_unit};')
    logger.info(f'  Validate every {valid_period} {record_unit};')
    logger.info(f'  Report every {report_period} {record_unit}.')

    for epoch in range(1, life_cycle + 1):
        model.train()
        tic = time.time()
        for step, data in enumerate(train_dataloader, start=1):
            data: Data = data.to(device_descriptor)
            optimizer.zero_grad()
            output, gnn_pooling_loss = model(data.x, data.edge_index, data.y[:, 0], data.batch)
            main_loss = criterion(output, data.y[:, 1])
            loss = main_loss + gnn_pooling_loss
            loss.backward()
            optimizer.step()

            average_losses = torch.tensor([float(loss), float(main_loss), float(gnn_pooling_loss)]).to(device_descriptor)
            average_loss_names = ['Total-Loss', 'Regression-Loss (MSE)', 'Cluster-loss']

            period_operation(
                rank, master_rank, world_size,
                epoch, step, train_period, valid_period, report_period, start_position,
                record_unit, 'Step',
                model, optimizer, checkpoint_dirpath, checkpoint_name, keep_number,
                valid_dataloader,
                average_losses, average_loss_names,
                device_descriptor,
                is_distribution
            )

        toc = time.time()
        logger.info(f'  Epoch@{epoch+start_position} Finished. Time Cost = {toc-tic:.2f}s')
        period_operation(
            is_master, world_size,
            epoch, step, train_period, valid_period, report_period, start_position,
            record_unit, 'Epoch',
            model, optimizer, checkpoint_dirpath, checkpoint_name, keep_number,
            average_losses, average_loss_names,
            is_distribution
        )
    
    if is_distribution:
        distributed.destroy_process_group()


def train(
    dataset_dirpath: pathlib.Path,
    mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised',
    x_feature_get_type: Literal['OnlyOp'] = 'OnlyOp',
    y_feature_get_type: Literal['OnlyMt'] = 'OnlyMt',

    node_dim: int = 512,
    metric_dim: int = 512,
    hidden_dim: int = 512,
    readout_dim: int = 256,
    cluster_num: int | None = None,

    checkpoint_filepath: str | None = None,
    checkpoint_dirpath: str = 'checkpoints',
    checkpoint_name: str = 'checkpoint',
    keep_number: int = 50,
    reset_optimizer: bool = True,
    reset_period: bool = True,
    fine_tune: bool = False,

    life_cycle: int = 100,
    train_period: int = 1000,
    valid_period: int = 1000,
    report_period: int = 100,
    record_unit: Literal['Epoch', 'Step'] = 'Step',

    train_batch_size: int = 32,
    valid_batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-1,

    device: Literal['CPU', 'GPU'] = 'GPU',
    world_size: int = 1,
    master_addr: str = 'localhost',
    master_port: str = '16161',
    master_rank: int = 0,

    seed: int = 1234,
):
    assert mode in {'Supervised', 'Unsupervised'}
    assert record_unit in {'Epoch', 'Step'}
    assert device in {'CPU', 'GPU'}

    if device == 'CPU':
        is_distribution = False
    if device == 'GPU':
        assert torch.cuda.device_count() >= world_size, f'Insufficient GPU: {torch.cuda.device_count()}'
        assert master_rank < world_size, f'Wrong Master Rank: {master_rank}'
        is_distribution = False if world_size == 1 else True
    logger.info(f'Using Device: {device};')
    logger.info(f'Distribution: {is_distribution}; {f"(Total {world_size} GPU)" if is_distribution else ""}')

    logger.info(f'Preparing ...')
    logger.info(f'  Mode: {mode}')
    logger.info(f'  v Building Younger Datasets ...')

    if mode == 'Supervised':
        train_dataset = YoungerDataset(dataset_dirpath, mode=mode, split='Train', x_feature_get_type=x_feature_get_type, y_feature_get_type=y_feature_get_type)
        valid_dataset = YoungerDataset(dataset_dirpath, mode=mode, split='Valid', x_feature_get_type=x_feature_get_type, y_feature_get_type=y_feature_get_type)

    if mode == 'Unsupervised':
        dataset = YoungerDataset(dataset_dirpath, mode=mode, x_feature_get_type=x_feature_get_type, y_feature_get_type=y_feature_get_type)
        dataset_size = len(dataset)

        random_generator = numpy.random.default_rng(seed=seed)
        random_index = list(random_generator.permutation(dataset_size))
        dataset = dataset[random_index]

        train_dataset = dataset[                    : dataset_size // 12 ]
        valid_dataset = dataset[ dataset_size // 12 :                    ]

    node_dict = train_dataset.node_dict
    metric_dict = train_dataset.metric_dict
    
    logger.info(f'    -> Node Dict Size: {len(node_dict)}')
    logger.info(f'    -> Metric Dict Size: {len(metric_dict)}')
    logger.info(f'    -> Dataset Split Sizes:')
    logger.info(f'       Train - {len(train_dataset)}')
    logger.info(f'       Valid - {len(valid_dataset)}')
    logger.info(f'  ^ Built.')

    if cluster_num is None:
        cluster_num = infer_cluster_num(dataset)
        logger.info(f'  Cluster Number Not Specified! Infered Number: {cluster_num}')
    else:
        logger.info(f'  Cluster Number: {cluster_num}')

    logger.info(f'  v Building Younger Model ...')
    model = NAPPGNNBase(
        node_dict=node_dict,
        metric_dict=metric_dict,
        node_dim=node_dim,
        metric_dim=metric_dim,
        hidden_dim=hidden_dim,
        readout_dim=readout_dim,
        cluster_num=cluster_num,
        mode=mode
    )

    parameters_number = get_model_parameters_number(model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    logger.info(
        f'\n  - Model Architecture:'
        f'\n{model}'
        f'\n  - Number of Parameters:'
        f'\n{parameters_number_str}'
        f'\n  ^ Built.'
    )

    if is_distribution:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        torch.multiprocessing.spawn(
            exact_train,
            args=(
                model, train_dataset, valid_dataset,
                checkpoint_filepath, checkpoint_dirpath, checkpoint_name, keep_number, reset_optimizer, reset_period, fine_tune,
                life_cycle, train_period, valid_period, report_period, record_unit,
                train_batch_size, valid_batch_size, learning_rate, weight_decay,
                seed, device, world_size, master_rank, is_distribution,
            ),
            nprocs=world_size,
            join=True
        )
    else:
        exact_train(0,
            model, train_dataset, valid_dataset,
            checkpoint_filepath, checkpoint_dirpath, checkpoint_name, keep_number, reset_optimizer, reset_period, fine_tune,
            life_cycle, train_period, valid_period, report_period, record_unit,
            train_batch_size, valid_batch_size, learning_rate, weight_decay,
            seed, device, world_size, master_rank, is_distribution,
        )


def test(
    dataset_dirpath: pathlib.Path,
    x_feature_get_type: Literal['OnlyOp'] = 'OnlyOp',
    y_feature_get_type: Literal['OnlyMt'] = 'OnlyMt',

    checkpoint_filepath: str | None = None,
    test_batch_size: int = 32,

    node_dim: int = 512,
    metric_dim: int = 512,
    hidden_dim: int = 512,
    readout_dim: int = 256,
    cluster_num: int | None = None,

    device: Literal['CPU', 'GPU'] = 'GPU',
):
    assert device in {'CPU', 'GPU'}
    device_descriptor = get_device_descriptor(device, 0)
    assert torch.cuda.is_available() or device == 'CPU'

    logger.info(f'Using Device: {device};')

    logger.info(f'  v Building Younger Datasets (Supervised)...')
    test_dataset = YoungerDataset(dataset_dirpath, mode='Supervised', split='Test', x_feature_get_type=x_feature_get_type, y_feature_get_type=y_feature_get_type)
    node_dict = test_dataset.node_dict
    metric_dict = test_dataset.metric_dict
    logger.info(f'    -> Node Dict Size: {len(node_dict)}')
    logger.info(f'    -> Metric Dict Size: {len(metric_dict)}')
    logger.info(f'    -> Test Dataset Size: {len(test_dataset)}')
    logger.info(f'  ^ Built.')

    logger.info(f'  v Building Younger Model ...')
    model = NAPPGNNBase(
        node_dict=node_dict,
        metric_dict=metric_dict,
        node_dim=node_dim,
        metric_dim=metric_dim,
        hidden_dim=hidden_dim,
        readout_dim=readout_dim,
        cluster_num=cluster_num,
        mode='Supervised'
    )

    parameters_number = get_model_parameters_number(model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    logger.info(
        f'\n  - Model Architecture:'
        f'\n{model}'
        f'\n  - Number of Parameters:'
        f'\n{parameters_number_str}'
        f'\n  ^ Built.'
    )

    logger.info(f'  v Loading Model Weights From Checkpoint [\'{checkpoint_filepath}\']...')
    checkpoint = load_checkpoint(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    logger.info(f'  ^ Loaded ')

    logger.info(f'  v Moving model to the specified device ...')
    model.to(device_descriptor)
    logger.info(f'  ^ Moved.')

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
    exact_check(device_descriptor, model, test_dataloader, 'Test')