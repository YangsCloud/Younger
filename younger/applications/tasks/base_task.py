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
import torch.utils.data

from typing import Any, Callable, Literal
from collections import OrderedDict

from younger.commons.logging import set_logger
from younger.commons.constants import YoungerHandle


class YoungerTask(object):
    def __init__(self, custom_config: dict) -> None:
        logging_config = dict()
        custom_logging_config = custom_config.get('logging', dict())
        logging_config['name'] = custom_logging_config.get('name', YoungerHandle.ApplicationsName + '-Task-' + 'Default')
        logging_config['mode'] = custom_logging_config.get('mode', 'console')
        logging_config['level'] = custom_logging_config.get('level', 'INFO')
        logging_config['filepath'] = custom_logging_config.get('filepath', None)
        self._logging_config = logging_config
        self._device_descriptor = None
        self._logger = None

    @property
    def device_descriptor(self):
        return self._device_descriptor

    @property
    def logger(self):
        if self._logger:
            logger = self._logger
        else:
            self._logger = set_logger(self._logging_config['name'], mode=self._logging_config['mode'], level=self._logging_config['level'], logging_filepath=self._logging_config['filepath'])
            logger = self._logger
        return logger

    def to(self, device_descriptor):
        self._device_descriptor = device_descriptor

    def reset(self):
        self._device_descriptor = None
        self._logger = None

    def update_learning_rate(self, stage: Literal['Step', 'Epoch'], **kwargs):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        return

    @property
    def model(self):
        raise NotImplementedError

    @model.setter
    def model(self, model):
        raise NotImplementedError

    @property
    def optimizer(self):
        raise NotImplementedError

    @property
    def train_dataset(self):
        raise NotImplementedError

    @property
    def valid_dataset(self):
        raise NotImplementedError

    @property
    def test_dataset(self):
        raise NotImplementedError

    def train(self, minibatch: Any) -> tuple[torch.Tensor, OrderedDict[str, tuple[torch.Tensor, Callable | None]]]:
        raise NotImplementedError

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict[str, tuple[torch.Tensor, Callable | None]]:
        raise NotImplementedError

    def prepare_cli(self):
        self.logger.warning(f'Not Implemented!')

    def cli(self, **kwargs):
        self.logger.warning(f'Not Implemented!')

    def api(self, **kwargs):
        self.logger.warning(f'Not Implemented!')