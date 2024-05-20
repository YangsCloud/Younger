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

from typing import Any
from collections import OrderedDict


class YoungerTask(object):
    def __init__(self, logger) -> None:
        self.logger = logger

    def train(self, minibatch: Any) -> tuple[torch.Tensor, OrderedDict]:
        raise NotImplementedError

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def eval_caculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict:
        raise NotImplementedError

    def api(self, **kwargs):
        raise NotImplementedError