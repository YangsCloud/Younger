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
import pathlib

from torch import distributed
from typing import Literal

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from younger.commons.io import create_dir
from younger.commons.logging import logger

from younger.applications.utils.neural_network import get_model_parameters_number, get_device_descriptor, fix_random_procedure, set_deterministic, load_checkpoint, save_checkpoint
from younger.applications.performance_prediction.models import NAPPGNNBase
from younger.applications.performance_prediction.datasets import YoungerDataset


def infer_cluster_num(dataset: Dataset) -> int:
    total_node_num = 0
    for data in dataset:
        total_node_num += data.num_nodes
    return total_node_num / len(dataset)


def get_logging_metrics_str(metric_names: list[str], metric_values: list[str]) -> str:
    metrics_str = str()
    for metric_name, metric_value in zip(metric_names, metric_values):
        metrics_str += f' {metric_name}: [{metric_value}]'
    return metrics_str


def period_operation(
    checkpoint_dirpath: pathlib.Path, mode: Literal['Supervised', 'Unsupervised'],
    is_master: bool, world_size: int,
    epoch: int, step: int, train_period: int, valid_period: int, report_period: int,
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_name: str, keep_number: int,
    valid_dataloader: DataLoader | None,
    digits: torch.Tensor, digit_names: list[str],
    device_descriptor: torch.device,
    is_distribution: bool
):
    if step % report_period == 0:
        if is_distribution:
            distributed.all_reduce(digits, op = distributed.ReduceOp.SUM)
            digits = digits / world_size
        digits = [f'{float(digit):.4f}' for digit in digits]
        logger.info(f'  [Epoch/Step]@[{epoch}/{step}] -{get_logging_metrics_str(digit_names, digits)}')

    if step % train_period == 0 and is_master:
        logger.info('  -> Saving checkpoint ...')
        tic = time.time()
        checkpoint = dict()
        checkpoint['Epoch'] = epoch
        checkpoint['Step'] = step
        checkpoint['model_state'] = model.state_dict()
        checkpoint['optimizer_state'] = optimizer.state_dict()
        save_checkpoint(checkpoint, checkpoint_path=checkpoint_dirpath, checkpoint_name=checkpoint_name, keep_number=keep_number)
        toc = time.time()
        logger.info(f'  -> Checkpoint is saved to \'{checkpoint_dirpath}\' at [Epoch/Step][{epoch}/{step}] (Time Cost: {toc-tic:.2f}s)')        

    if step % valid_period == 0:
        if is_distribution:
            distributed.barrier()
        exact_check(device_descriptor, model, valid_dataloader, 'Valid', mode, is_distribution, world_size)
        if is_distribution:
            distributed.barrier()
        model.train()


def exact_check(device_descriptor: torch.device, model: torch.nn.Module, dataloader: DataLoader, split: Literal['Valid', 'Test'], mode: Literal['Supervised', 'Unsupervised'], is_distribution: bool = False, world_size: int = 1):
    model.eval()
    logger.info(f'  -> {split} Begin ...')
    tic = time.time()
    if mode == 'Supervised':
        digit_names = ['ACC', 'MAE', 'RMSE']
        overall_mae = 0
        overall_mse = 0

        overall_correct = 0
        overall_predict = 0
    else:
        digit_names = ['Cluster-loss']
        overall_global_pooling_loss = 0
    with torch.no_grad():
        for index, data in enumerate(dataloader, start=1):
            data: Data = data.to(device_descriptor)
            if mode == 'Supervised':
                y = data.y.reshape(len(data), -1)
                cls_output, reg_output, _ = model(data.x, data.edge_index, data.batch)
                mae = torch.nn.functional.l1_loss(reg_output.reshape(-1), y[:, 1], reduction='sum')
                mse = torch.nn.functional.mse_loss(reg_output.reshape(-1), y[:, 1], reduction='sum')
                overall_mae += mae
                overall_mse += mse

                cls_predict = torch.argmax(torch.nn.functional.softmax(cls_output, dim=-1), dim=-1)
                cls_correct = (cls_predict == y[:, 0].long()).sum().item()
                overall_correct += cls_correct
                overall_predict += len(cls_predict)
            else:
                global_pooling_loss = model(data.x, data.edge_index, data.batch)
                overall_global_pooling_loss += global_pooling_loss
    toc = time.time()

    if mode == 'Supervised':
        digits = torch.tensor([
            overall_correct/overall_predict,
            overall_mae/overall_predict,
            torch.sqrt(overall_mse/overall_predict),
        ]).to(device_descriptor)
    else:
        digits = torch.tensor([overall_global_pooling_loss/index]).to(device_descriptor)

    if is_distribution:
        distributed.all_reduce(digits, op = distributed.ReduceOp.SUM)
        digits = digits / world_size

    digits = [f'{float(digit):.4f}' for digit in digits]

    logger.info(f'  -> {split} Finished. Overall Result - {get_logging_metrics_str(digit_names, digits)} (Time Cost = {toc-tic:.2f}s)')


def exact_train(
    rank: int,
    checkpoint_dirpath: pathlib.Path, mode: Literal['Supervised', 'Unsupervised'],
    model: torch.nn.Module, train_dataset: Dataset, valid_dataset: Dataset,
    checkpoint_filepath: str, checkpoint_name: str, keep_number: int, reset_optimizer: bool, reset_period: bool, fine_tune: bool,
    life_cycle:int, update_period: int, train_period: int, valid_period: int, report_period: int,
    train_batch_size: int, valid_batch_size: int, learning_rate: float, weight_decay: float, shuffle: bool,
    seed: int, device: Literal['CPU', 'GPU'], world_size: int, master_rank: int, is_distribution: bool,
    make_deterministic: bool,
):
    is_master = rank == master_rank
    fix_random_procedure(seed)
    set_deterministic(make_deterministic)
    device_descriptor = get_device_descriptor(device, rank)

    model.to(device_descriptor)
    logger.info(f'  Model Moved to Device \'{device_descriptor}\'')

    if is_master:
        logger.disabled = False
    else:
        logger.disabled = True

    if is_distribution:
        distributed.init_process_group('nccl', rank=rank, world_size=world_size)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if is_distribution:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed, drop_last=False)
    else:
        train_sampler = RandomSampler(train_dataset) if shuffle else None
        valid_sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, sampler=valid_sampler)

    if checkpoint_filepath:
        checkpoint = load_checkpoint(pathlib.Path(checkpoint_filepath), checkpoint_name)
    else:
        checkpoint = None

    if checkpoint is None:
        logger.info(f'  Train from scratch.')
        start_position = 0
    else:
        logger.info(f'  Train from checkpoint [\'{checkpoint_filepath}\'] [Epoch/Step]@[{checkpoint["Epoch"]}/{checkpoint["Step"]}].')

        if reset_optimizer:
            logger.info(f'  Reset Optimizer.')
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        logger.info(f'  v Loading Parameters ...')
        model.load_state_dict(checkpoint['model_state'])
        logger.info(f'  ^ Loaded.')

        if reset_period:
            logger.info(f'  Reset Epoch & Step.')
            start_position = 0
        else:
            start_position = checkpoint['Step']

    logger.info(f'Training Start ...')
    logger.info(f'  Train Life Cycle: Total {life_cycle} Epochs!')
    logger.info(f'  Update every {update_period} Step;')
    logger.info(f'  Report every {report_period} Step;')
    logger.info(f'  Validate every {valid_period} Step;')
    logger.info(f'  Saving checkpoint every {train_period} Step.')

    cre_criterian = torch.nn.CrossEntropyLoss()
    mse_criterian = torch.nn.MSELoss()
    model.train()
    optimizer.zero_grad()
    epoch = 0
    step = start_position
    while epoch < life_cycle:
        epoch += 1
        tic = time.time()
        for data in train_dataloader:
            step += 1
            data: Data = data.to(device_descriptor)
            if mode == 'Supervised':
                y = data.y.reshape(len(data), -1)
                cls_output, reg_output, global_pooling_loss = model(data.x, data.edge_index, data.batch)
                # Now Try Classification
                cls_main_loss = cre_criterian(torch.nn.functional.softmax(cls_output, dim=-1), y[:, 0].long())
                # Now Try Regression
                reg_main_loss = mse_criterian(reg_output.reshape(-1), y[:, 1])
                loss = cls_main_loss + reg_main_loss + global_pooling_loss
                average_digits = torch.tensor([float(loss), float(cls_main_loss), float(reg_main_loss), float(global_pooling_loss)]).to(device_descriptor) / update_period
                average_digit_names = ['Total-Loss', 'CLS-Loss (CRE)', 'REG-Loss (MSE)', 'Cluster-loss']
                loss = loss / update_period
                loss.backward()
            else:
                global_pooling_loss = model(data.x, data.edge_index, data.batch)
                average_digits = torch.tensor([float(global_pooling_loss)]).to(device_descriptor) / update_period
                average_digit_names = ['Cluster-loss']
                global_pooling_loss = global_pooling_loss / update_period
                global_pooling_loss.backward()

            if step % update_period == 0:
                optimizer.step()
                optimizer.zero_grad()

            period_operation(
                checkpoint_dirpath, mode,
                is_master, world_size,
                epoch, step, train_period, valid_period, report_period,
                model, optimizer, checkpoint_name, keep_number,
                valid_dataloader,
                average_digits, average_digit_names,
                device_descriptor,
                is_distribution
            )

        toc = time.time()
        logger.info(f'  <=> Epoch@{epoch} Finished. Time Cost = {toc-tic:.2f}s')

    if is_distribution:
        distributed.destroy_process_group()


def train(
    train_dataset_dirpath: pathlib.Path,
    valid_dataset_dirpath: pathlib.Path,
    checkpoint_dirpath: pathlib.Path,
    mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised',
    x_feature_get_type: Literal['OnlyOp'] = 'OnlyOp',
    y_feature_get_type: Literal['OnlyMt'] = 'OnlyMt',

    node_dim: int = 512,
    hidden_dim: int = 512,
    readout_dim: int = 256,
    cluster_num: int | None = None,

    checkpoint_filepath: str | None = None,
    checkpoint_name: str = 'checkpoint',
    keep_number: int = 50,
    reset_optimizer: bool = True,
    reset_period: bool = True,
    fine_tune: bool = False,

    life_cycle: int = 100,
    update_period: int = 1,
    train_period: int = 1000,
    valid_period: int = 1000,
    report_period: int = 100,

    train_batch_size: int = 32,
    valid_batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-1,
    shuffle: bool = True,

    device: Literal['CPU', 'GPU'] = 'GPU',
    world_size: int = 1,
    master_addr: str = 'localhost',
    master_port: str = '16161',
    master_rank: int = 0,
    worker_number: int = 4,

    seed: int = 1234,
    make_deterministic: bool = False,
):
    assert mode in {'Supervised', 'Unsupervised'}
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

    fix_random_procedure(seed)
    train_dataset = YoungerDataset(
        str(train_dataset_dirpath),
        x_feature_get_type=x_feature_get_type,
        y_feature_get_type=y_feature_get_type,
        worker_number=worker_number
    )
    valid_dataset = YoungerDataset(
        str(valid_dataset_dirpath),
        x_feature_get_type=x_feature_get_type,
        y_feature_get_type=y_feature_get_type,
        worker_number=worker_number
    )

    meta = train_dataset.meta

    logger.info(f'    -> Node Dict Size: {len(meta["o2i"])}')
    if mode == 'Supervised':
        logger.info(f'    -> Task Number: {len(meta["t2i"])}')
        logger.info(f'    -> Dataset Number: {len(meta["d2i"])}')
        logger.info(f'    -> Metric Number: {len(meta["m2i"])}')
    logger.info(f'    -> Dataset Split Sizes:')
    logger.info(f'       Train - {len(train_dataset)}')
    logger.info(f'       Valid - {len(valid_dataset)}')
    logger.info(f'  ^ Datasets Built.')

    if cluster_num is None:
        cluster_num = infer_cluster_num(train_dataset)
        logger.info(f'  Cluster Number Not Specified! Infered Number: {cluster_num}')
    else:
        logger.info(f'  Cluster Number: {cluster_num}')

    logger.info(f'  v Building Younger Model ...')
    model = NAPPGNNBase(
        meta=meta,
        node_dim=node_dim,
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
    )
    logger.info(f'  ^ Model Built.')

    create_dir(checkpoint_dirpath)
    if is_distribution:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        torch.multiprocessing.spawn(
            exact_train,
            args=(
                checkpoint_dirpath, mode,
                model, train_dataset, valid_dataset,
                checkpoint_filepath, checkpoint_name, keep_number, reset_optimizer, reset_period, fine_tune,
                life_cycle, update_period, train_period, valid_period, report_period,
                train_batch_size, valid_batch_size, learning_rate, weight_decay, shuffle,
                seed, device, world_size, master_rank, is_distribution,
                make_deterministic,
            ),
            nprocs=world_size,
            join=True
        )
    else:
        exact_train(0,
            checkpoint_dirpath, mode,
            model, train_dataset, valid_dataset,
            checkpoint_filepath, checkpoint_name, keep_number, reset_optimizer, reset_period, fine_tune,
            life_cycle, update_period, train_period, valid_period, report_period,
            train_batch_size, valid_batch_size, learning_rate, weight_decay, shuffle,
            seed, device, world_size, master_rank, is_distribution,
            make_deterministic,
        )


def test(
    test_dataset_dirpath: pathlib.Path,
    checkpoint_filepath: pathlib.Path,
    x_feature_get_type: Literal['OnlyOp'] = 'OnlyOp',
    y_feature_get_type: Literal['OnlyMt'] = 'OnlyMt',

    test_batch_size: int = 32,

    node_dim: int = 512,
    hidden_dim: int = 512,
    readout_dim: int = 256,
    cluster_num: int | None = None,

    device: Literal['CPU', 'GPU'] = 'GPU',
    worker_number: int = 4,
):
    assert device in {'CPU', 'GPU'}
    device_descriptor = get_device_descriptor(device, 0)
    assert torch.cuda.is_available() or device == 'CPU'

    logger.info(f'Using Device: {device};')

    logger.info(f'  v Building Younger Datasets (Supervised)...')
    test_dataset = YoungerDataset(
        str(test_dataset_dirpath),
        x_feature_get_type=x_feature_get_type,
        y_feature_get_type=y_feature_get_type,
        worker_number=worker_number
    )
    meta = test_dataset.meta
    logger.info(f'    -> Node Dict Size: {len(meta["o2i"])}')
    logger.info(f'    -> Task Number: {len(meta["t2i"])}')
    logger.info(f'    -> Dataset Number: {len(meta["d2i"])}')
    logger.info(f'    -> Metric Number: {len(meta["m2i"])}')
    logger.info(f'    -> Test Dataset Size: {len(test_dataset)}')
    logger.info(f'  ^ Built.')

    logger.info(f'  v Building Younger Model ...')
    model = NAPPGNNBase(
        meta=meta,
        node_dim=node_dim,
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
    exact_check(device_descriptor, model, test_dataloader, 'Test', 'Supervised')
