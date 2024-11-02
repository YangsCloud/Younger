#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-07 17:23
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import torch
import numpy
import random
import pathlib

from typing import Any, Iterable, Literal
from numpy.typing import NDArray

from younger.commons.io import save_json, save_pickle, load_json, load_pickle


def set_deterministic(make_deterministic: bool = True):
    if make_deterministic:
        torch.use_deterministic_algorithms(True)


def shuffled(sequence: Iterable) -> Iterable:
    indices = list(range(len(sequence)))
    random.shuffle(indices)
    shuffled_sequence = ( sequence[index] for index in indices )
    return shuffled_sequence


def fix_random_procedure(seed: int):
    assert 0 < seed, 'Seed must > 0 .'

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model_parameters_number(model: torch.nn.Module) -> int:
    parameters_number = dict()
    for name, parameters in model.named_parameters():
        root_name = name.split('.')[0]
        if root_name in parameters_number:
            parameters_number[root_name] += parameters.numel()
        else:
            parameters_number[root_name] = parameters.numel()

    return parameters_number


def get_device_descriptor(device: str, index: int) -> torch.device:
    if device == 'CPU':
        device_name = 'cpu'

    if device == 'GPU':
        device_name = f'cuda:{index}'

    return torch.device(device_name)


def find_all_checkpoints(checkpoint_dirpath: pathlib.Path, checkpoint_name: str = 'checkpoint') -> dict[int, pathlib.Path]:
    checkpoint_filename_pattern = re.compile(f'{checkpoint_name}_Epoch_(?:\d+)_Step_(\d+)\.cp')
    checkpoints = dict()
    for path in checkpoint_dirpath.iterdir():
        if path.is_file():
            result = checkpoint_filename_pattern.fullmatch(path.name)
            if result is not None:
                position = int(result.group(1))
                checkpoints[position] = path
            else:
                continue
        else:
            continue

    return checkpoints


def load_checkpoint(checkpoint_path: pathlib.Path, checkpoint_name: str = 'checkpoint') -> dict[str, Any] | None:
    checkpoint = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if checkpoint_path.is_dir():
        assert len(checkpoint_name) != 0, f'Invalid checkpoint name.'
        checkpoints = find_all_checkpoints(checkpoint_path, checkpoint_name)

        if len(checkpoints) == 0:
            latest_checkpoint = None
        else:
            max_position = max(checkpoints.keys())
            latest_checkpoint_path = checkpoints[max_position]
            if latest_checkpoint_path.is_file():
                latest_checkpoint = torch.load(latest_checkpoint_path, map_location=torch.device('cpu'))
                assert max_position == latest_checkpoint['Step'], 'An Error occurred when loading checkpoint.'
            else:
                latest_checkpoint = None

        checkpoint = latest_checkpoint

    return checkpoint


def save_checkpoint(checkpoint, checkpoint_path: pathlib.Path, checkpoint_name: str = 'checkpoint', keep_number: int = 1):
    if checkpoint_path.is_dir():
        assert len(checkpoint_name) != 0, f'Invalid checkpoint name.'
        position = checkpoint['Step']
        checkpoint_filename = f'{checkpoint_name}_Epoch_{checkpoint["Epoch"]}_Step_{checkpoint["Step"]}.cp'
        checkpoint_filepath = checkpoint_path.joinpath(checkpoint_filename)
        torch.save(checkpoint, checkpoint_filepath)

        checkpoints = find_all_checkpoints(checkpoint_path, checkpoint_name)
        positions = sorted(list(checkpoints.keys()), reverse=True)
        for position in positions[keep_number:]:
            remove_checkpoint(checkpoints[position])
    else:
        checkpoint_filepath = checkpoint_path
        torch.save(checkpoint, checkpoint_filepath)


def remove_checkpoint(checkpoint_path: pathlib.Path):
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    else:
        raise IOError(f'Invalid address: {checkpoint_path}')


def save_operator_embedding(save_dirpath: pathlib.Path, weights: NDArray, op_dict: dict[str, int]):
    weights_filepath = save_dirpath.joinpath(f'weights.npy')
    op_dict_filepath = save_dirpath.joinpath(f'op_dict.json')
    numpy.save(weights_filepath, weights)
    save_json(op_dict, op_dict_filepath, indent=2)


def load_operator_embedding(load_dirpath: pathlib.Path):
    weights_filepath = load_dirpath.joinpath(f'weights.npy')
    op_dict_filepath = load_dirpath.joinpath(f'op_dict.json')
    weights = numpy.load(weights_filepath)
    op_dict = load_json(op_dict_filepath)
    return weights, op_dict