#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-16 23:43
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import torch
import torch.utils.data

from typing import Literal
from collections import OrderedDict
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GAE, VGAE

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from younger.commons.logging import Logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.models import SAGE_NP
from younger.applications.datasets import EgoDataset
from younger.applications.tasks.base_task import YoungerTask
from younger.applications.utils.neural_network import load_checkpoint


class NodeEmbedding(YoungerTask):
    def __init__(self, custom_config: dict) -> None:
        super().__init__(custom_config)
        self.build_config(custom_config)
        self._model = None
        self._optimizer = None
        self._train_dataset = None

    def build_config(self, custom_config: dict):
        mode = custom_config.get('mode', 'Train')
        assert mode in {'Train', 'Test', 'API'}

        # Dataset
        dataset_config = dict()
        custom_dataset_config = custom_config.get('dataset', dict())
        dataset_config['train_dataset_dirpath'] = custom_dataset_config.get('train_dataset_dirpath', None)
        dataset_config['worker_number'] = custom_dataset_config.get('worker_number', 4)

        # Model
        model_config = dict()
        custom_model_config = custom_config.get('model', dict())
        model_config['node_dim'] = custom_model_config.get('node_dim', 512)
        model_config['hidden_dim'] = custom_model_config.get('hidden_dim', 256)
        model_config['dropout'] = custom_model_config.get('dropout', 0.5)

        # Optimizer
        optimizer_config = dict()
        custom_optimizer_config = custom_config.get('optimizer', dict())
        optimizer_config['learning_rate'] = custom_optimizer_config.get('learning_rate', 0.01)
        optimizer_config['weight_decay'] = custom_optimizer_config.get('weight_decay', 5e-4)

        # Scheduler
        scheduler_config = dict()
        custom_scheduler_config = custom_config.get('scheduler', dict())
        scheduler_config['factor'] = custom_scheduler_config.get('factor', 0.2)
        scheduler_config['min_lr'] = custom_scheduler_config.get('min_lr', 5e-5)

        # API
        api_config = dict()
        custom_api_config = custom_config.get('api', dict())
        api_config['meta_filepath'] = custom_api_config.get('meta_filepath', None)
        api_config['onnx_model_dirpath'] = custom_api_config.get('onnx_model_dirpath', list())

        config = dict()
        config['dataset'] = dataset_config
        config['model'] = model_config
        config['optimizer'] = optimizer_config
        config['scheduler'] = scheduler_config
        config['api'] = api_config
        config['mode'] = mode
        self.config = config

    @property
    def train_dataset(self):
        if self._train_dataset:
            train_dataset = self._train_dataset
        else:
            if self.config['mode'] == 'Train':
                self._train_dataset = EgoDataset(
                    self.config['dataset']['train_dataset_dirpath'],
                    'train',
                    worker_number=self.config['dataset']['worker_number'],
                )

                self.logger.info(f'    -> Number of Top Operators: {len(self._train_dataset.x_dict["o2i"] - 3)}')
                self.top_dict_size = len(self._train_dataset.x_dict["o2i"])
                self.logger.info(f'    -> Number of Long Tail Operators: {len(self._train_dataset.x_dict["lto"])}')
                self.lt_dict_size = len(self._train_dataset.x_dict["lto"])
            else:
                self._train_dataset = None
            train_dataset = self._train_dataset
        return train_dataset

    @property
    def valid_dataset(self):
        return None

    @property
    def test_dataset(self):
        return None

    @property
    def model(self):
        if self._model:
            model = self._model
        else:
            self.logger.info(f"    -> Using Model: GraphSAGE")
            self._model = SAGE_NP(
                node_dict_size=self.top_dict_size,
                node_dim=self.config['model']['node_dim'],
                hidden_dim=self.config['model']['hidden_dim'],
                dropout=self.config['model']['dropout'],
            )
            model = self._model
        return model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def optimizer(self):
        if self._optimizer:
            optimizer = self._optimizer
        else:
            if self.config['mode'] == 'Train':
                self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer']['learning_rate'], weight_decay=self.config['optimizer']['weight_decay'])
                self._learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.config['scheduler']['factor'], min_lr=self.config['scheduler']['min_lr'])
            else:
                self._optimizer = None
            optimizer = self._optimizer
        return optimizer

    def update_learning_rate(self, stage: Literal['Step', 'Epoch'], **kwargs):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        if stage == 'Epoch':
            # self._learning_rate_scheduler.step(kwargs['loss'])
            pass
        return

    def train(self, minibatch: Data) -> tuple[torch.Tensor, OrderedDict]:
        minibatch = minibatch.to(self.device_descriptor)
        output = self.model(minibatch.x, minibatch.edge_index, minibatch.mask_x_position)
        loss = torch.nn.functional.nll_loss(output, minibatch.mask_x_label)

        logs = OrderedDict({
            'loss': (loss, lambda x: f'{x:.4f}'),
        })
        return loss, logs

    def eval(self, minibatch: Data) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict:
        pass

    def api(self, device_descriptor, **kwargs):
        pass