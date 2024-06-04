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


from cProfile import label
import torch
import torch.utils.data

from typing import Any, Literal
from collections import OrderedDict
from torch_geometric.data import Batch, Data

from younger.commons.logging import Logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.models import GLASS
from younger.applications.datasets import BlockDataset
from younger.applications.tasks.base_task import YoungerTask


class BlockEmbedding(YoungerTask):
    def __init__(self, custom_config: dict) -> None:
        super().__init__(custom_config)
        self.build_config(custom_config)
        self._model = None
        self._optimizer = None
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    def build_config(self, custom_config: dict):
        mode = custom_config.get('mode', 'Train')
        assert mode in {'Train', 'Test', 'API'}

        # Dataset
        dataset_config = dict()
        custom_dataset_config = custom_config.get('dataset', dict())
        dataset_config['train_dataset_dirpath'] = custom_dataset_config.get('train_dataset_dirpath', None)
        dataset_config['valid_dataset_dirpath'] = custom_dataset_config.get('valid_dataset_dirpath', None)
        dataset_config['test_dataset_dirpath'] = custom_dataset_config.get('test_dataset_dirpath', None)
        dataset_config['block_get_type'] = custom_dataset_config.get('block_get_type', None)
        dataset_config['block_get_number'] = custom_dataset_config.get('block_get_number', None)
        dataset_config['encode_type'] = custom_dataset_config.get('encode_type', 'node')
        dataset_config['seed'] = custom_dataset_config.get('seed', None)
        dataset_config['worker_number'] = custom_dataset_config.get('worker_number', 4)
        dataset_config['node_dict_size'] = custom_dataset_config.get('node_dict_size', None)

        # Model
        model_config = dict()
        custom_model_config = custom_config.get('model', dict())
        model_config['hidden_dim'] = custom_model_config.get('hidden_dim', 64)
        model_config['output_dim'] = custom_model_config.get('output_dim', 1)
        model_config['aggr_type'] = custom_model_config.get('aggr_type', 'avg')
        model_config['pool_type'] = custom_model_config.get('pool_type', 'mean')
        model_config['dropout'] = custom_model_config.get('dropout', 0.2)
        model_config['ratio'] = custom_model_config.get('ratio', 0.8)
        model_config['label'] = custom_model_config.get('label', 'coreness')

        # Optimizer
        optimizer_config = dict()
        custom_optimizer_config = custom_config.get('optimizer', dict())
        optimizer_config['learning_rate'] = custom_optimizer_config.get('learning_rate', 0.0005)
        optimizer_config['weight_decay'] = custom_optimizer_config.get('weight_decay', 0)

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
                self._train_dataset = BlockDataset(
                    self.config['dataset']['train_dataset_dirpath'],
                    node_dict_size=self.config['dataset']['node_dict_size'],
                    worker_number=self.config['dataset']['worker_number'],
                    block_get_type=self.config['dataset']['block_get_type'],
                    encode_type=self.config['dataset']['encode_type'],
                    seed=self.config['dataset']['seed']
                )
                if self.config['dataset']['encode_type'] == 'node':
                    self.logger.info(f'    -> Nodes Dict Size: {len(self._train_dataset.x_dict["n2i"])}')
                    self.node_dict_size = len(self._train_dataset.x_dict["n2i"])
                elif self.config['dataset']['encode_type'] == 'operator':
                    self.logger.info(f'    -> Nodes Dict Size: {len(self._train_dataset.x_dict["o2i"])}')
                    self.node_dict_size = len(self._train_dataset.x_dict["o2i"])
            else:
                self._train_dataset = None
            train_dataset = self._train_dataset
        return train_dataset

    @property
    def valid_dataset(self):
        if self._valid_dataset:
            valid_dataset = self._valid_dataset
        else:
            self._valid_dataset = None
            valid_dataset = self._valid_dataset
        return valid_dataset

    @property
    def test_dataset(self):
        if self._test_dataset:
            test_dataset = self._test_dataset
        else:
            if self.config['mode'] == 'Test':
                self._test_dataset = BlockDataset(
                    self.config['dataset']['test_dataset_dirpath'],
                    node_dict_size=self.config['dataset']['node_dict_size'],
                    worker_number=self.config['dataset']['worker_number'],
                    block_get_type=self.config['dataset']['block_get_type'],
                    encode_type=self.config['dataset']['encode_type'],
                    block_get_number=self.config['dataset']['block_get_number'],
                    seed=self.config['dataset']['seed']
                )
                if self.config['dataset']['encode_type'] == 'node':
                    self.logger.info(f'    -> Nodes Dict Size: {len(self._test_dataset.x_dict["n2i"])}')
                    self.node_dict_size = len(self._test_dataset.x_dict["n2i"])
                elif self.config['dataset']['encode_type'] == 'operator':
                    self.logger.info(f'    -> Nodes Dict Size: {len(self._test_dataset.x_dict["o2i"])}')
                    self.node_dict_size = len(self._test_dataset.x_dict["o2i"])
            else:
                self._test_dataset = None
            test_dataset = self._test_dataset
        return test_dataset

    @property
    def model(self):
        if self._model:
            model = self._model
        else:
            self._model = GLASS(
                node_dict_size=self.node_dict_size,
                hidden_dim=self.config['model']['hidden_dim'],
                output_dim=self.config['model']['output_dim'],
                pool_type=self.config['model']['pool_type'],
                dropout=self.config['model']['dropout'],
                aggr_type=self.config['model']['aggr_type'],
                ratio=self.config['model']['ratio'],
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
                label_name_to_id = dict(
                    density = 0,
                    coreness = 1,
                    dnc = [0, 1],
                    cut_ratio = 2,
                )

                self.label_id = label_name_to_id[self.config['model']['label']]
            else:
                self._optimizer = None
            optimizer = self._optimizer
        return optimizer

    def update_learning_rate(self, stage: Literal['Step', 'Epoch'], **kwargs):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        if stage == 'Epoch':
            self._learning_rate_scheduler.step(kwargs['loss'])
        return

    def train(self, minibatch: Data) -> tuple[torch.Tensor, OrderedDict]:
        minibatch = minibatch.to(self.device_descriptor)
        subgraph_label = minibatch.block_labels.reshape(len(minibatch), -1)[:, self.label_id]
        output = self.model(minibatch.x, minibatch.edge_index, minibatch.edge_attr, minibatch.block_mask, minibatch.batch)
        loss = torch.nn.functional.mse_loss(output, subgraph_label)
        logs = OrderedDict({
            'REG-Loss (MSE)': (loss, lambda x: f'{x:.4f}'),
        })
        return loss, logs

    def eval(self, minibatch: Data) -> tuple[torch.Tensor, torch.Tensor]:
        minibatch = minibatch.to(self.device_descriptor)
        # Return Output & Golden
        subgraph_label = minibatch.block_labels.reshape(len(minibatch), -1)[:, self.label_id]
        outputs = self.model(minibatch.x, minibatch.edge_index, minibatch.edge_attr, minibatch.block_mask, minibatch.batch)
        return outputs.reshape(-1), subgraph_label

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict:
        all_outputs = torch.cat(all_outputs)
        all_goldens = torch.cat(all_goldens)
        mae = torch.nn.functional.l1_loss(all_outputs, all_goldens, reduction='mean')
        mse = torch.nn.functional.mse_loss(all_outputs, all_goldens, reduction='mean')
        rmse = torch.sqrt(mse)
        logs = OrderedDict({
            'MAE': (mae, lambda x: f'{x:.4f}'),
            'MSE': (mse, lambda x: f'{x:.4f}'),
            'RMSE': (rmse, lambda x: f'{x:.4f}'),
        })
        return logs

    def api(self, device_descriptor, **kwargs):
        meta_filepath = self.config['api']['meta_filepath']
        onnx_model_dirpath = self.config['api']['onnx_model_dirpath']
        assert meta_filepath, f'No Meta File.'
        assert onnx_model_dirpath, f'No ONNX Dir.'

        self.logger.info(f'  v Loading Meta ...')
        meta = BlockDataset.load_meta(meta_filepath)
        x_dict = BlockDataset.get_x_dict(meta, node_dict_size=self.config['dataset']['node_dict_size'])
        y_dict = BlockDataset.get_y_dict(meta, task_dict_size=self.config['dataset']['task_dict_size'])
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
            data = BlockDataset.get_data(standardized_graph, x_dict, y_dict, feature_get_type='none')
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
