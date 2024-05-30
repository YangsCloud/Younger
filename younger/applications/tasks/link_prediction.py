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
from torch_geometric.data import Batch, Data
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.commons.logging import set_logger
from younger.commons.constants import YoungerHandle
from younger.commons.logging import Logger

from younger.applications.models import GCN_LP, GAT_LP, SAGE_LP
from younger.applications.datasets import LinkDataset
from younger.applications.tasks.base_task import YoungerTask


class LinkPridiction(YoungerTask):
    def __init__(self, custom_config: dict, device_descriptor: torch.device) -> None:
        super().__init__(custom_config, device_descriptor)
        self.build_config(custom_config)
        self.build()

    def build_config(self, custom_config: dict):
        mode = custom_config.get('mode', 'Train')
        assert mode in {'Train', 'Test', 'API'}

        # Dataset
        dataset_config = dict()
        custom_dataset_config = custom_config.get('dataset', dict())
        dataset_config['train_dataset_dirpath'] = custom_dataset_config.get('train_dataset_dirpath', None)
        dataset_config['valid_dataset_dirpath'] = custom_dataset_config.get('valid_dataset_dirpath', None)
        dataset_config['test_dataset_dirpath'] = custom_dataset_config.get('test_dataset_dirpath', None)
        dataset_config['link_get_number'] = custom_dataset_config.get('link_get_number', None)
        dataset_config['worker_number'] = custom_dataset_config.get('worker_number', 4)
        dataset_config['seed'] = custom_dataset_config.get('seed', None)

        # Model
        model_config = dict()
        custom_model_config = custom_config.get('model', dict())
        model_config['node_dim'] = custom_model_config.get('node_dim', 256)
        model_config['hidden_dim'] = custom_model_config.get('hidden_dim', 128)
        model_config['output_dim'] = custom_model_config.get('output_dim', 64)

        # Optimizer
        optimizer_config = dict()
        custom_optimizer_config = custom_config.get('optimizer', dict())
        optimizer_config['learning_rate'] = custom_optimizer_config.get('learning_rate', 0.0005)
        optimizer_config['weight_decay'] = custom_optimizer_config.get('weight_decay', 0)

        # Scheduler
        scheduler_config = dict()
        custom_scheduler_config = custom_config.get('scheduler', dict())
        scheduler_config['step_size'] = custom_scheduler_config.get('step_size', 100)
        scheduler_config['gamma'] = custom_scheduler_config.get('gamma', 0.5)

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

    def build(self):
        if self.config['mode'] == 'Train':
            self._train_dataset = LinkDataset(
                self.config['dataset']['train_dataset_dirpath'],
                worker_number=self.config['dataset']['worker_number'],
                seed=self.config['dataset']['seed']
            )
            self._valid_dataset = LinkDataset(
                self.config['dataset']['valid_dataset_dirpath'],
                link_get_number=self.config['dataset']['link_get_number'],
                worker_number=self.config['dataset']['worker_number'],
                seed=self.config['dataset']['seed']
            )
            self.logger.info(f'    -> Nodes Dict Size: {len(self.train_dataset.x_dict["n2i"])}')
            self.node_dict_size = len(self.train_dataset.x_dict["n2i"])

        if self.config['mode'] == 'Test':
            self._test_dataset = LinkDataset(
                self.config['dataset']['test_dataset_dirpath'],
                link_get_number=self.config['dataset']['link_get_number'],
                worker_number=self.config['dataset']['worker_number']
            )
            self.node_dict_size = len(self.test_dataset.x_dict["n2i"])

        self._model = GAT_LP(
            node_dict_size=self.node_dict_size,
            node_dim=self.config['model']['node_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            output_dim=self.config['model']['output_dim'],
        )

        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer']['learning_rate'], weight_decay=self.config['optimizer']['weight_decay'])
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.config['scheduler']['step_size'], gamma=self.config['scheduler']['gamma'])
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, factor=self.config['scheduler']['factor'], min_lr=self.config['scheduler']['min_lr'])

    
    def update_learning_rate(self, stage: Literal['Step', 'Epoch'], **kwargs):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        if stage == 'Epoch':
            # self.learning_rate_scheduler.step()
            pass
        return

    def train(self, minibatch: Any) -> tuple[torch.Tensor, OrderedDict]:
        minibatch, step = minibatch
        minibatch = minibatch.to(self.device_descriptor)
        output = self.model(minibatch, minibatch.link)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output.reshape(-1), minibatch.link_label)
        logs = OrderedDict({
            'loss': (loss, lambda x: f'{x:.4f}')
        })

        return loss, logs

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        minibatch = minibatch.to(self.device_descriptor)
        # Return Output & Golden
        link_label = minibatch.link_label
        output = self.model(minibatch, minibatch.link).view(-1).sigmoid()
    
        return output, link_label

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict:

        all_outputs = torch.cat(all_outputs).reshape(-1).cpu().numpy()
        all_goldens = torch.cat(all_goldens).reshape(-1).cpu().numpy()

        pred = (all_outputs > 0.5).astype(int)
        f1 = f1_score(all_goldens, pred)
        ap = average_precision_score(all_goldens, all_outputs)
        auc = roc_auc_score(all_goldens, all_outputs)
        logs = OrderedDict({
            'AUC': (auc, lambda x: f'{x:.4f}'),
            'F1': (f1, lambda x: f'{x:.4f}'),
            'AP': (ap, lambda x: f'{x:.4f}'),
        })
        
        return logs

    def api(self, device_descriptor, **kwargs):
        meta_filepath = self.config['api']['meta_filepath']
        onnx_model_dirpath = self.config['api']['onnx_model_dirpath']
        assert meta_filepath, f'No Meta File.'
        assert onnx_model_dirpath, f'No ONNX Dir.'

        self.logger.info(f'  v Loading Meta ...')
        meta = LinkDataset.load_meta(meta_filepath)
        x_dict = LinkDataset.get_x_dict(meta, node_dict_size=self.config['dataset']['node_dict_size'])
        y_dict = LinkDataset.get_y_dict(meta, task_dict_size=self.config['dataset']['task_dict_size'])
        self.logger.info(f'    -> Tasks Dict Size: {len(x_dict)}')
        self.logger.info(f'    -> Nodes Dict Size: {len(y_dict)}')
        self.logger.info(f'  ^ Built.')

        self.logger.info(f'  v Loading ONNX Models')
        datas = list()
        onnx_model_filenames = list()
        for onnx_model_filepath in onnx_model_dirpath.iterdir():
            onnx_model_filenames.append(onnx_model_filepath.name)
            instance = Instance(onnx_model_filepath)
            standardized_graph = Network.standardize(instance.network.graph)
            for node_index in standardized_graph.nodes():
                operator = standardized_graph.nodes[node_index]['features']['operator']
                attributes = standardized_graph.nodes[node_index]['features']['attributes']
                standardized_graph.nodes[node_index]['features']['attributes'] = get_complete_attributes_of_node(attributes, operator['op_type'], operator['domain'], meta['max_inclusive_version'])
            standardized_graph.graph.clear()
            data = LinkDataset.get_data(standardized_graph, x_dict, y_dict, feature_get_type='none')
            datas.append(data)
        minibatch = Batch.from_data_list(datas)
        self.logger.info(f'  ^ Loaded. Total - {len(datas)}.')

        self.model.eval()
        self.logger.info(f'  -> Interact Test Begin ...')

        with torch.no_grad():
            minibatch: Data = minibatch.to(device_descriptor)
            output, _ = self.model(minibatch.x, minibatch.edge_index, minibatch.batch)

            for onnx_model_filename, output_value in zip(onnx_model_filenames, output):
                self.logger.info(f'  -> Result - {onnx_model_filename}: {output_value}')

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset