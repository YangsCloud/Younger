#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-19 18:30
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch
import pathlib
import torch.utils.data

from typing import Any, Callable, Literal
from collections import OrderedDict

from younger.commons.logging import set_logger
from younger.commons.constants import YoungerHandle


class YoungerTask(object):
    def __init__(self, custom_config: dict, device_descriptor: torch.device) -> None:
        logging_config = dict()
        custom_logging_config = custom_config.get('logging', dict())
        logging_config['name'] = custom_logging_config.get('name', YoungerHandle.ApplicationsName + '-Task-' + 'Default')
        logging_config['mode'] = custom_logging_config.get('mode', 'console')
        logging_config['level'] = custom_logging_config.get('level', 'INFO')
        logging_config['filepath'] = custom_logging_config.get('filepath', None)
        self.logging_config = logging_config
        self.logger = set_logger(logging_config['name'], mode=logging_config['mode'], level=logging_config['level'], logging_filepath=logging_config['filepath'])
        self.device_descriptor = device_descriptor

    def update_learning_rate(self, stage: Literal['Step', 'Epoch']):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        return

    def train(self, minibatch: Any) -> tuple[torch.Tensor, OrderedDict[str, tuple[torch.Tensor, Callable | None]]]:
        raise NotImplementedError

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict[str, tuple[torch.Tensor, Callable | None]]:
        raise NotImplementedError

    def api(self, **kwargs):
        raise NotImplementedError