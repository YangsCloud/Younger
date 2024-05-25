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
    def __init__(self, logger: Logger, custom_config: dict) -> None:
        super().__init__(logger)
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
        dataset_config['worker_number'] = custom_dataset_config.get('worker_number', 4)
        dataset_config['node_dict_size'] = custom_dataset_config.get('node_dict_size', None)
        dataset_config['task_dict_size'] = custom_dataset_config.get('task_dict_size', None)

        # Model
        model_config = dict()
        custom_model_config = custom_config.get('model', dict())
        model_config['node_dim'] = custom_model_config.get('node_dim', 512)
        model_config['task_dim'] = custom_model_config.get('task_dim', 512)
        model_config['hidden_dim'] = custom_model_config.get('hidden_dim', 512)
        model_config['readout_dim'] = custom_model_config.get('readout_dim', 256)
        model_config['cluster_num'] = custom_model_config.get('cluster_num', None)

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

    def build(self):
        if self.config['mode'] == 'Train':
            self.train_dataset = BlockDataset(
                self.config['dataset']['train_dataset_dirpath'],
                task_dict_size=self.config['dataset']['task_dict_size'],
                node_dict_size=self.config['dataset']['node_dict_size'],
                worker_number=self.config['dataset']['worker_number']
            )
            self.valid_dataset = BlockDataset(
                self.config['dataset']['valid_dataset_dirpath'],
                task_dict_size=self.config['dataset']['task_dict_size'],
                node_dict_size=self.config['dataset']['node_dict_size'],
                worker_number=self.config['dataset']['worker_number']
            )
            self.logger.info(f'    -> Nodes Dict Size: {len(self.train_dataset.x_dict["n2i"])}')
            self.logger.info(f'    -> Tasks Dict Size: {len(self.train_dataset.y_dict["t2i"])}')

        if self.config['mode'] == 'Test':
            self.test_dataset = BlockDataset(
                self.config['dataset']['test_dataset_dirpath'],
                task_dict_size=self.config['dataset']['task_dict_size'],
                node_dict_size=self.config['dataset']['node_dict_size'],
                worker_number=self.config['dataset']['worker_number']
            )

        self.model = GLASS(
            node_dict=self.train_dataset.x_dict['n2i'],
            task_dict=self.train_dataset.y_dict['t2i'],
            node_dim=self.config['model']['node_dim'],
            task_dim=self.config['model']['task_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            readout_dim=self.config['model']['readout_dim'],
            cluster_num=self.config['model']['cluster_num'],
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer']['learning_rate'], weight_decay=self.config['optimizer']['weight_decay'])
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.config['scheduler']['factor'], min_lr=self.config['scheduler']['min_lr'])

    def update_learning_rate(self, stage: Literal['Step', 'Epoch']):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        if stage == 'Epoch':
            self.learning_rate_scheduler.step()
        return

    def train(self, minibatch: Any) -> tuple[torch.Tensor, OrderedDict]:
        y = minibatch.y.reshape(len(minibatch), -1)
        tasks = y[:, 0].long().clone()
        global_pooling_loss = self.model(minibatch.x[:, 0].clone(), minibatch.edge_index, minibatch.batch)
        logs = OrderedDict({
            'Cluster-loss': float(global_pooling_loss),
        })
        return global_pooling_loss, logs

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # Return Output & Golden
        y = minibatch.y.reshape(len(minibatch), -1)
        tasks = y[:, 0].long()
        outputs, _ = self.model(minibatch.x, minibatch.edge_index, minibatch.batch)
        return outputs, None

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict:
        all_outputs = torch.mean(torch.stack(all_outputs))
        logs = OrderedDict({
            'Cluster-loss': float(logs),
        })
        return logs

    def api(self, device_descriptor, **kwargs):
        meta_filepath = self.config['api']['meta_filepath']
        onnx_model_dirpath = self.config['api']['onnx_model_dirpath']
        assert meta_filepath, f'No Meta File.'
        assert onnx_model_dirpath, f'No ONNX Dir.'

        self.logger.info(f'  v Loading Meta ...')
        meta = ArchitectureDataset.load_meta(meta_filepath)
        x_dict = ArchitectureDataset.get_x_dict(meta, node_dict_size=self.config['dataset']['node_dict_size'])
        y_dict = ArchitectureDataset.get_y_dict(meta, task_dict_size=self.config['dataset']['task_dict_size'])
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
            data = ArchitectureDataset.get_data(standardized_graph, x_dict, y_dict, feature_get_type='none')
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