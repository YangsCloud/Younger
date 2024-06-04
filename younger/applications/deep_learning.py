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
import tqdm
import time
import torch
import pathlib

from torch import distributed
from typing import Literal
from collections import OrderedDict

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader

from younger.commons.io import create_dir, load_toml

from younger.applications.utils.neural_network import get_model_parameters_number, get_device_descriptor, fix_random_procedure, set_deterministic, load_checkpoint, save_checkpoint

from younger.applications.tasks import task_builders, YoungerTask


def get_logging_metrics_str(metrics: OrderedDict[str, str]) -> str:
    metrics_str = str()
    for metric_name, metric_value in metrics.items():
        metrics_str += f' [{metric_name}]={metric_value}'
    return metrics_str


def exact_eval(
    task: YoungerTask,
    dataloader: DataLoader, 
    split: Literal['Valid', 'Test'],
):
    task.logger.info(f'-> {split} Begin ...')
    all_outputs = list()
    all_goldens = list()
    tic = time.time()
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as progress_bar:
            for index, minibatch in enumerate(dataloader, start=1):
                eval_result = task.eval(minibatch)
                if eval_result is None:
                    pass
                else:
                    outputs, goldens = eval_result

                    all_outputs.append(outputs)
                    all_goldens.append(goldens)
                progress_bar.update(1)
    toc = time.time()

    logs = task.eval_calculate_logs(all_outputs, all_goldens)
    if logs is None:
        task.logger.info(f'-> {split} Finished. No User Defined Output')
    else:
        metrics = OrderedDict()
        for log_key, (log_value, log_format) in logs.items():
            metrics[log_key] = log_format(float(log_value))
        task.logger.info(f'-> {split} Finished. Overall Result -{get_logging_metrics_str(metrics)} (Time Cost = {toc-tic:.2f}s)')


def exact_train(
    rank: int,
    distribution_flag: bool,
    master_rank: int,
    world_size: int,
    seed: int, make_deterministic: bool,
    task: YoungerTask, config_filepath: pathlib.Path,

    checkpoint_dirpath: pathlib.Path, checkpoint_name: str, keep_number: int,

    train_batch_size: int, valid_batch_size: int, shuffle: bool,

    checkpoint_filepath: pathlib.Path, reset_optimizer: bool, reset_period: bool,

    life_cycle:int, report_period: int, update_period: int, train_period: int, valid_period: int,

    device: Literal['CPU', 'GPU'],
):
    device_descriptor = get_device_descriptor(device, rank)

    task.reset()
    task.to(device_descriptor)
    task.model.to(device_descriptor)
    task.logger.info(f'Model Moved to Device \'{device_descriptor}\'')

    is_master = rank == master_rank
    fix_random_procedure(seed)
    set_deterministic(make_deterministic)
    torch.autograd.set_detect_anomaly(True)

    if is_master:
        task.logger.disabled = False
    else:
        task.logger.disabled = True

    task.logger.info(f'Using Device: {device};')
    task.logger.info(f'Distribution: {distribution_flag}; {f"(Total {world_size} GPU)" if distribution_flag else ""}')

    create_dir(checkpoint_dirpath)
    task.logger.info(f'Checkpoint will saved into: \'{checkpoint_dirpath}\'')

    # Build Model
    task.logger.info(f'Preparing Model ...')

    # Print Model
    task.logger.info(f'-> Model Specs:')
    parameters_number = get_model_parameters_number(task.model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    task.logger.info(
        f'\n======= v Model Architecture v ======='
        f'\n{task.model}'
        f'\n'
        f'\n====== v Number of Parameters v ======'
        f'\n{parameters_number_str}'
    )

    # Model
    if distribution_flag:
        distributed.init_process_group('nccl', rank=rank, world_size=world_size)
        task.model = torch.nn.parallel.DistributedDataParallel(task.model, device_ids=[rank], find_unused_parameters=False)

    # Datasets
    if distribution_flag:
        train_sampler = DistributedSampler(task.train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=True)
    else:
        train_sampler = RandomSampler(task.train_dataset) if shuffle else None
    train_dataloader = DataLoader(task.train_dataset, batch_size=train_batch_size, sampler=train_sampler)
    if task.valid_dataset:
        valid_dataloader = DataLoader(task.valid_dataset, batch_size=valid_batch_size, shuffle=False)
    else:
        valid_dataloader = None
    # Init Train Status
    if checkpoint_filepath:
        checkpoint = load_checkpoint(pathlib.Path(checkpoint_filepath), checkpoint_name)
    else:
        checkpoint = None

    if checkpoint is None:
        task.logger.info(f'Train from scratch.')
        start_position = 0
    else:
        task.logger.info(f'Train from checkpoint [\'{checkpoint_filepath}\'] [Epoch/Step]@[{checkpoint["Epoch"]}/{checkpoint["Step"]}].')

        if reset_optimizer:
            task.logger.info(f'Reset Optimizer.')
        else:
            task.optimizer.load_state_dict(checkpoint['optimizer_state'])

        task.logger.info(f'v Loading Parameters ...')
        task.model.load_state_dict(checkpoint['model_state'])
        task.logger.info(f'^ Loaded.')

        if reset_period:
            task.logger.info(f'Reset Epoch & Step.')
            start_position = 0
        else:
            start_position = checkpoint['Step']

    task.logger.info(f'-> Training Start ...')
    task.logger.info(f'  Train Life Cycle: Total {life_cycle} Epochs!')
    task.logger.info(f'  Update every {update_period} Step;')
    task.logger.info(f'  Report every {report_period} Step;')
    task.logger.info(f'  Validate every {valid_period} Step;')
    task.logger.info(f'  Saving checkpoint every {train_period} Step.')

    task.model.train()
    task.optimizer.zero_grad()
    epoch = 0
    step = start_position
    while epoch < life_cycle:
        if distribution_flag:
            train_sampler.set_epoch(epoch)
        epoch += 1

        tic = time.time()
        for minibatch in train_dataloader:
            step += 1
            (loss, logs) = task.train(minibatch)

            # Report Model Parameters
            if step % report_period == 0:
                metrics = OrderedDict()
                for log_key, (log_value, log_format) in logs.items():
                    if distribution_flag:
                        distributed.all_reduce(log_value, op = distributed.ReduceOp.SUM)
                        log_value = log_value / world_size
                    metrics[log_key] = log_format(float(log_value))
                task.logger.info(f'  [Epoch/Step]@[{epoch}/{step}] -{get_logging_metrics_str(metrics)}')

            # Update Model Parameters
            if step % update_period == 0:
                retain_graph = False
                task.optimizer.step()
                task.optimizer.zero_grad()
            else:
                retain_graph = True
            loss.backward(retain_graph=retain_graph)

            # Save Model Parameters
            if step % train_period == 0 and is_master:
                task.logger.info('-> Saving checkpoint ...')
                tic = time.time()
                checkpoint = dict()
                checkpoint['Epoch'] = epoch
                checkpoint['Step'] = step
                checkpoint['model_state'] = task.model.module.state_dict() if distribution_flag else task.model.state_dict()
                checkpoint['optimizer_state'] = task.optimizer.state_dict()
                save_checkpoint(checkpoint, checkpoint_path=checkpoint_dirpath, checkpoint_name=checkpoint_name, keep_number=keep_number)
                toc = time.time()
                task.logger.info(f'-> Checkpoint is saved to \'{checkpoint_dirpath}\' at [Epoch/Step][{epoch}/{step}] (Time Cost: {toc-tic:.2f}s)')        

            # Do Validation
            if step % valid_period == 0:
                task.model.eval()
                if distribution_flag:
                    distributed.barrier()
                if is_master:
                    if valid_dataloader:
                        exact_eval(
                            task,
                            valid_dataloader, 
                            'Valid',
                        )
                if distribution_flag:
                    distributed.barrier()
                task.model.train()

            task.update_learning_rate(stage='Step', step=step, epoch=epoch, loss=loss)

        toc = time.time()
        task.logger.info(f'-> Epoch@{epoch} Finished. Time Cost = {toc-tic:.2f}s')

        task.update_learning_rate(stage='Epoch', step=step, epoch=epoch, loss=loss)

    if distribution_flag:
        distributed.destroy_process_group()


def train(
    task_name: str, config_filepath: pathlib.Path,

    checkpoint_dirpath: pathlib.Path, checkpoint_name: str = 'checkpoint', keep_number: int = 50,

    train_batch_size: int = 32, valid_batch_size: int = 32, shuffle: bool = True,

    checkpoint_filepath: str | None = None, reset_optimizer: bool = True, reset_period: bool = True,

    life_cycle: int = 100, report_period: int = 100, update_period: int = 1, train_period: int = 1000, valid_period: int = 1000,

    device: Literal['CPU', 'GPU'] = 'GPU',
    world_size: int = 1, master_addr: str = 'localhost', master_port: str = '16161', master_rank: int = 0,
    seed: int = 1234, make_deterministic: bool = False,
):
    assert task_name in task_builders, f'Task ({task_name}) is not Defined'

    custom_config = load_toml(config_filepath)
    task: YoungerTask = task_builders[task_name](custom_config)
    task.logger.info(f'Configuration Loaded From {config_filepath}')

    task.logger.info(f'Task: \'{task_name}\'')

    task.logger.info(f'Preparing Datasets ...')
    # Print Dataset
    task.logger.info(f'Dataset Split Sizes:')
    task.logger.info(f' Train - {len(task.train_dataset)}')
    if task.valid_dataset:
        task.logger.info(f' Valid - {len(task.valid_dataset)}')

    assert device in {'CPU', 'GPU'}
    if device == 'CPU':
        distribution_flag = False
    if device == 'GPU':
        assert torch.cuda.device_count() >= world_size, f'Insufficient GPU: {torch.cuda.device_count()}'
        assert master_rank < world_size, f'Wrong Master Rank: {master_rank}'
        distribution_flag = False if world_size == 1 else True

    if distribution_flag:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        torch.multiprocessing.spawn(
            exact_train,
            args=(
                distribution_flag,
                master_rank,
                world_size,
                seed, make_deterministic,

                task, config_filepath,

                checkpoint_dirpath, checkpoint_name, keep_number,

                train_batch_size, valid_batch_size, shuffle,

                checkpoint_filepath, reset_optimizer, reset_period,

                life_cycle, report_period, update_period, train_period, valid_period,

                device,
            ),
            nprocs=world_size,
            join=True
        )
    else:
        exact_train(0,
            distribution_flag,
            master_rank,
            world_size,
            seed, make_deterministic,

            task, config_filepath,

            checkpoint_dirpath, checkpoint_name, keep_number,

            train_batch_size, valid_batch_size, shuffle,

            checkpoint_filepath, reset_optimizer, reset_period,

            life_cycle, report_period, update_period, train_period, valid_period,

            device,
        )


def test(
    task_name: str, config_filepath: pathlib.Path,
    checkpoint_filepath: pathlib.Path,
    test_batch_size: int = 32,
    device: Literal['CPU', 'GPU'] = 'GPU',
):
    assert task_name in task_builders, f'Task ({task_name}) is not Defined'

    assert device in {'CPU', 'GPU'}
    device_descriptor = get_device_descriptor(device, 0)
    assert torch.cuda.is_available() or device == 'CPU'

    # Build Task
    custom_config = load_toml(config_filepath)
    task: YoungerTask = task_builders[task_name](custom_config)
    task.logger.info(f'Configuration Loaded From {config_filepath}')

    task.logger.info(f'Using Device: {device};')

    task.logger.info(f'Preparing Task: Model & Dataset ...')
    # Print Dataset
    task.logger.info(f'-> Dataset Split Size:')
    task.logger.info(f'   Test - {len(task.test_dataset)}')

    # Print Model
    task.logger.info(f'-> Model Specs:')
    parameters_number = get_model_parameters_number(task.model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    task.logger.info(
        f'\n======= v Model Architecture v ======='
        f'\n{task.model}'
        f'\n'
        f'\n====== v Number of Parameters v ======'
        f'\n{parameters_number_str}'
    )

    task.logger.info(f'v Loading Model Weights From Checkpoint [\'{checkpoint_filepath}\']...')
    checkpoint = load_checkpoint(checkpoint_filepath)
    task.model.load_state_dict(checkpoint['model_state'], strict=True)
    task.logger.info(f'^ Loaded ')

    task.logger.info(f'v Moving model to the specified device ...')
    task.model.to(device_descriptor)
    task.logger.info(f'^ Moved.')

    test_dataloader = DataLoader(task.test_dataset, batch_size=test_batch_size, shuffle=False)
    exact_eval(
        task,
        test_dataloader, 
        'Test',
    )

def api(
    task_name: str, config_filepath: pathlib.Path,
    checkpoint_filepath: pathlib.Path,
    device: Literal['CPU', 'GPU'] = 'GPU',
    **kwargs,
):
    # TODO: Serve!
    assert task_name in task_builders, f'Task ({task_name}) is not Defined'

    assert device in {'CPU', 'GPU'}
    device_descriptor = get_device_descriptor(device, 0)
    assert torch.cuda.is_available() or device == 'CPU'

    # Build Task
    custom_config = load_toml(config_filepath)
    task: YoungerTask = task_builders[task_name](custom_config)
    task.logger.info(f'Configuration Loaded From {config_filepath}')

    task.logger.info(f'Using Device: {device};')

    task.logger.info(f'Preparing Task: Model & Dataset ...')
    # Print Model
    task.logger.info(f'  -> Model Specs:')
    parameters_number = get_model_parameters_number(task.model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    task.logger.info(
        f'\n  - Model Architecture:'
        f'\n{task.model}'
        f'\n  - Number of Parameters:'
        f'\n{parameters_number_str}'
    )

    task.logger.info(f'  v Loading Model Weights From Checkpoint [\'{checkpoint_filepath}\']...')
    checkpoint = load_checkpoint(checkpoint_filepath)
    
    task.model.load_state_dict(checkpoint['model_state'], strict=True)
    task.logger.info(f'  ^ Loaded ')

    task.logger.info(f'  v Moving model to the specified device ...')
    task.model.to(device_descriptor)
    task.logger.info(f'  ^ Moved.')

    tic = time.time()
    task.api(**kwargs)
    toc = time.time()

    task.logger.info(f'  -> Test Finished. (Time Cost = {toc-tic:.2f}s)')