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

import numpy
import pathlib

import torch
import torch.utils.data

from typing import Any, Literal
from collections import OrderedDict
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GAE, VGAE

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.models import GCN_NP, GIN_NP, GAT_NP, SAGE_NP, Encoder_NP, LinearCls
from younger.applications.datasets import NodeDataset
from younger.applications.tasks.base_task import YoungerTask
from younger.applications.utils.neural_network import load_checkpoint


class NodePrediction(YoungerTask):
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
        dataset_config['dataset_name'] = custom_dataset_config.get('dataset_name', 'Younger_NP')
        dataset_config['encode_type'] = custom_dataset_config.get('encode_type', 'node')
        dataset_config['standard_onnx'] = custom_dataset_config.get('standard_onnx', False)
        dataset_config['worker_number'] = custom_dataset_config.get('worker_number', 4)

        # Model
        model_config = dict()
        custom_model_config = custom_config.get('model', dict())
        model_config["model_type"] = custom_model_config.get('model_type', None)
        model_config['node_dim'] = custom_model_config.get('node_dim', 512)
        model_config['hidden_dim'] = custom_model_config.get('hidden_dim', 256)
        model_config['layer_number'] = custom_model_config.get('layer_number', 3)
        model_config['dropout'] = custom_model_config.get('dropout', 0.5)
        model_config['stage'] = custom_model_config.get('stage', None) # This is for VGAE or GAE
        model_config['ae_type'] = custom_model_config.get('ae_type', 'VGAE') # This is for VGAE or GAE
        model_config['emb_checkpoint_path'] = custom_model_config.get('emb_checkpoint_path', None) # This is for VGAE or GAE

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

        # Embedding
        embedding_config = dict()
        custom_embedding_config = custom_config.get('embedding', dict())
        embedding_config['activate'] = custom_embedding_config.get('activate', False)
        embedding_config['embedding_dirpath'] = custom_embedding_config.get('embedding_dirpath', None)

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
        config['embedding'] = embedding_config
        config['api'] = api_config
        config['mode'] = mode
        self.config = config

    @property
    def train_dataset(self):
        if self._train_dataset:
            train_dataset = self._train_dataset
        else:
            if self.config['mode'] == 'Train':
                self._train_dataset = NodeDataset(
                    self.config['dataset']['train_dataset_dirpath'],
                    'train',
                    dataset_name=self.config['dataset']['dataset_name'],
                    encode_type=self.config['dataset']['encode_type'],
                    standard_onnx=self.config['dataset']['standard_onnx'],
                    worker_number=self.config['dataset']['worker_number'],
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
            if self.config['mode'] == 'Train':
                self._valid_dataset = NodeDataset(
                    self.config['dataset']['valid_dataset_dirpath'],
                    'valid',
                    dataset_name=self.config['dataset']['dataset_name'],
                    encode_type=self.config['dataset']['encode_type'],
                    standard_onnx=self.config['dataset']['standard_onnx'],
                    worker_number=self.config['dataset']['worker_number'],
                )
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
                self._test_dataset = NodeDataset(
                    self.config['dataset']['test_dataset_dirpath'],
                    'test',
                    dataset_name=self.config['dataset']['dataset_name'],
                    encode_type=self.config['dataset']['encode_type'],
                    standard_onnx=self.config['dataset']['standard_onnx'],
                    worker_number=self.config['dataset']['worker_number'],
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
            self.logger.info(f"    -> Using Model: {self.config['model']['model_type']}")
            if self.config['model']['model_type'] == 'SAGE_NP':
                self._model = SAGE_NP(
                    node_dict_size=self.node_dict_size,
                    node_dim=self.config['model']['node_dim'],
                    hidden_dim=self.config['model']['hidden_dim'],
                    dropout=self.config['model']['dropout'],
                )

            elif self.config['model']['model_type'] == 'GIN_NP':
                self._model = GIN_NP(
                    node_dict_size=self.node_dict_size,
                    node_dim=self.config['model']['node_dim'],
                    hidden_dim=self.config['model']['hidden_dim'],
                    dropout=self.config['model']['dropout'],
                    layer_number=self.config['model']['layer_number'],
                )

            elif self.config['model']['model_type'] == 'VGAE_NP':
                if self.config['model']['stage'] == 'encoder':
                    self._model = VGAE(Encoder_NP(
                        node_dict_size=self.node_dict_size,
                        node_dim=self.config['model']['node_dim'],
                        hidden_dim=self.config['model']['hidden_dim'],
                        ae_type=self.config['model']['ae_type'],
                    ))
                elif self.config['model']['stage'] == 'classification':
                    self._model = LinearCls(
                        node_dict_size=self.node_dict_size,
                        hidden_dim=self.config['model']['hidden_dim'],
                    )
                    checkpoint = load_checkpoint(pathlib.Path(self.config['model']['emb_checkpoint_path']))
                    self.ae_model= VGAE(Encoder_NP(
                        node_dict_size=self.node_dict_size,
                        node_dim=self.config['model']['node_dim'],
                        hidden_dim=self.config['model']['hidden_dim'],
                        ae_type=self.config['model']['ae_type'],
                    ))
                    self.ae_model.load_state_dict(checkpoint['model_state'])
                    self.ae_model.to(self.device_descriptor)

            elif self.config['model']['model_type'] == 'GAE_NP':
                if self.config['model']['stage'] == 'encoder':
                    self._model = GAE(Encoder_NP(
                        node_dict_size=self.node_dict_size,
                        node_dim=self.config['model']['node_dim'],
                        hidden_dim=self.config['model']['hidden_dim'],
                        ae_type=self.config['model']['ae_type'],
                    ))
                elif self.config['model']['stage'] == 'classification':
                    self._model = LinearCls(
                        node_dict_size=self.node_dict_size,
                        hidden_dim=self.config['model']['hidden_dim'],
                    )
                    checkpoint = load_checkpoint(pathlib.Path(self.config['model']['emb_checkpoint_path']))
                    self.ae_model= GAE(Encoder_NP(
                        node_dict_size=self.node_dict_size,
                        node_dim=self.config['model']['node_dim'],
                        hidden_dim=self.config['model']['hidden_dim'],
                        ae_type=self.config['model']['ae_type'],
                    ))
                    self.ae_model.load_state_dict(checkpoint['model_state'])
                    self.ae_model.to(self.device_descriptor)
                
            elif self.config['model']['model_type'] == 'GCN_NP':
                self._model = GCN_NP(
                    node_dict_size=self.node_dict_size,
                    node_dim=self.config['model']['node_dim'],
                    hidden_dim=self.config['model']['hidden_dim'],
                    dropout=self.config['model']['dropout'],
                )
            
            elif self.config['model']['model_type'] == 'GAT_NP':
                self._model = GAT_NP(
                    node_dict_size=self.node_dict_size,
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
        # The following code is for VGAE.
        if self.config['model']['stage'] == 'encoder':
            z = self.model.encode(minibatch.x, minibatch.edge_index)
            loss = self.model.recon_loss(z, minibatch.edge_index)
            if self.config['model']['model_type'] == 'VGAE_NP':
                loss = loss + 0.001 * self.model.kl_loss()

        elif self.config['model']['stage'] == 'classification':
            self.ae_model.eval()
            embeddings = self.ae_model.encode(minibatch.x, minibatch.edge_index).detach()
            output = self.model(embeddings, minibatch.mask_x_position)
            loss = torch.nn.functional.nll_loss(output, minibatch.mask_x_label)

        # This is for other methods
        else:      
            output = self.model(minibatch.x, minibatch.edge_index, minibatch.mask_x_position)
            loss = torch.nn.functional.nll_loss(output, minibatch.mask_x_label)
        
        logs = OrderedDict({
            'loss': (loss, lambda x: f'{x:.4f}'),
        })
        return loss, logs

    def eval(self, minibatch: Data) -> tuple[torch.Tensor, torch.Tensor]:
        minibatch = minibatch.to(self.device_descriptor)
        if self.config['model']['stage'] == 'encoder':
            return 
        if self.config['model']['stage'] == 'classification':
            self.ae_model.eval()
            embeddings = self.ae_model.encode(minibatch.x, minibatch.edge_index).detach()
            output = self.model(embeddings, minibatch.mask_x_position)
        else:
            output = self.model(minibatch.x, minibatch.edge_index, minibatch.mask_x_position)

        # Return Output & Golden
        return output, minibatch.mask_x_label

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict:
        if self.config['model']['stage'] == 'encoder':
            return

        all_outputs = torch.cat(all_outputs)
        all_goldens = torch.cat(all_goldens)

        pred = all_outputs.max(1)[1].cpu().numpy()
        gold = all_goldens.cpu().numpy()

        print("pred[:5]:", pred[:5])
        print("gold[:5]:", gold[:5])

        acc = accuracy_score(gold, pred)
        macro_p = precision_score(gold, pred, average='macro', zero_division=0)
        macro_r = recall_score(gold, pred, average='macro', zero_division=0)
        macro_f1 = f1_score(gold, pred, average='macro', zero_division=0)
        micro_f1 = f1_score(gold, pred, average='micro', zero_division=0)

        logs = OrderedDict({
            'acc': (acc, lambda x: f'{x:.4f}'),
            'macro_p': (macro_p, lambda x: f'{x:.4f}'),
            'macro_r': (macro_r, lambda x: f'{x:.4f}'),
            'macro_f1': (macro_f1, lambda x: f'{x:.4f}'),
            'micro_f1': (micro_f1, lambda x: f'{x:.4f}'),
        })
        return logs

    def api(self, device_descriptor, **kwargs):
        meta_filepath = self.config['api']['meta_filepath']
        onnx_model_dirpath = self.config['api']['onnx_model_dirpath']
        assert meta_filepath, f'No Meta File.'
        assert onnx_model_dirpath, f'No ONNX Dir.'

        self.logger.info(f'  v Loading Meta ...')
        meta = NodeDataset.load_meta(meta_filepath)
        x_dict = NodeDataset.get_x_dict(meta, node_dict_size=self.config['dataset']['node_dict_size'])
        y_dict = NodeDataset.get_y_dict(meta, task_dict_size=self.config['dataset']['task_dict_size'])
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
            data = NodeDataset.get_data(standardized_graph, x_dict, y_dict, feature_get_type='none')
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
