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
import networkx

import torch
import torch.utils.data

from typing import Literal
from collections import OrderedDict
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from younger.commons.io import load_pickle
from younger.datasets.modules import Instance, Network, Dataset
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.models import MAEGIN
from younger.applications.datasets import SSLDataset
from younger.applications.tasks.base_task import YoungerTask
from younger.applications.utils.neural_network import load_checkpoint, save_pickle


class SSLPrediction(YoungerTask):
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
        assert mode in {'Train', 'Test', 'CLI', 'API'}

        # Dataset
        dataset_config = dict()
        custom_dataset_config = custom_config.get('dataset', dict())
        dataset_config['train_dataset_dirpath'] = custom_dataset_config.get('train_dataset_dirpath', None)
        dataset_config['valid_dataset_dirpath'] = custom_dataset_config.get('valid_dataset_dirpath', None)
        dataset_config['test_dataset_dirpath'] = custom_dataset_config.get('test_dataset_dirpath', None)
        dataset_config['dataset_name'] = custom_dataset_config.get('dataset_name', 'SSLDataset')
        dataset_config['encode_type'] = custom_dataset_config.get('encode_type', 'node')
        dataset_config['standard_onnx'] = custom_dataset_config.get('standard_onnx', False)
        dataset_config['worker_number'] = custom_dataset_config.get('worker_number', 4)
        dataset_config['mask_ratio'] = custom_dataset_config.get('mask_ratio', 0.15)

        # Model
        model_config = dict()
        custom_model_config = custom_config.get('model', dict())
        model_config["model_type"] = custom_model_config.get('model_type', MAEGIN)
        model_config['node_dim'] = custom_model_config.get('node_dim', 512)
        model_config['hidden_dim'] = custom_model_config.get('hidden_dim', 256)
        model_config['layer_number'] = custom_model_config.get('layer_number', 3)
        model_config['dropout'] = custom_model_config.get('dropout', 0.5)

        # Optimizer
        optimizer_config = dict()
        custom_optimizer_config = custom_config.get('optimizer', dict())
        optimizer_config['lr'] = custom_optimizer_config.get('lr', 0.001)
        optimizer_config['eps'] = custom_optimizer_config.get('eps', 1e-8)
        optimizer_config['weight_decay'] = custom_optimizer_config.get('weight_decay', 0.01)
        optimizer_config['amsgrad'] = custom_optimizer_config.get('amsgrad', False)

        # Scheduler
        scheduler_config = dict()
        custom_scheduler_config = custom_config.get('scheduler', dict())
        scheduler_config['start_factor'] = custom_scheduler_config.get('start_factor', 0.1)
        scheduler_config['warmup_steps'] = custom_scheduler_config.get('warmup_steps', 1500)
        scheduler_config['total_steps'] = custom_scheduler_config.get('total_steps', 150000)
        scheduler_config['last_step'] = custom_scheduler_config.get('last_step', -1)

        # CLI
        cli_config = dict()
        custom_cli_config = custom_config.get('cli', dict())
        cli_config['input_type'] = custom_cli_config.get('input_type', 'instance')
        cli_config['node_size_limit'] = custom_cli_config.get('node_size_limit', 4)
        cli_config['meta_filepath'] = custom_cli_config.get('meta_filepath', None)
        cli_config['result_filepath'] = custom_cli_config.get('result_filepath', None)
        cli_config['instances_dirpath'] = custom_cli_config.get('instances_dirpath', None)
        cli_config['subgraphs_dirpath'] = custom_cli_config.get('instances_dirpath', None)

        config = dict()
        config['dataset'] = dataset_config
        config['model'] = model_config
        config['optimizer'] = optimizer_config
        config['scheduler'] = scheduler_config
        config['cli'] = cli_config
        config['mode'] = mode
        self.config = config

    @property
    def train_dataset(self):
        if self._train_dataset:
            train_dataset = self._train_dataset
        else:
            if self.config['mode'] == 'Train':
                self._train_dataset = SSLDataset(
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
                self._valid_dataset = SSLDataset(
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
                self._test_dataset = SSLDataset(
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
            if self.config['model']['model_type'] == 'MAEGIN':
                self._model = MAEGIN(
                    node_dict_size=self.node_dict_size,
                    node_dim=self.config['model']['node_dim'],
                    hidden_dim=self.config['model']['hidden_dim'],
                    dropout=self.config['model']['dropout'],
                    layer_number=self.config['model']['layer_number'],
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
                self._optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['optimizer']['lr'], eps=self.config['optimizer']['eps'], weight_decay=self.config['optimizer']['weight_decay'], amsgrad=self.config['optimizer']['amsgrad'])
                # self._learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.config['scheduler']['factor'], min_lr=self.config['scheduler']['min_lr'])
                warmup_lr_schr = torch.optim.lr_scheduler.LinearLR(
                    self._optimizer,
                    start_factor=self.config['scheduler']['start_factor'],
                    total_iters=self.config['scheduler']['warmup_steps'],
                    last_epoch=self.config['scheduler']['last_step'],
                )
                cosine_lr_schr = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self._optimizer,
                    T_max=self.config['scheduler']['total_steps'] - self.config['scheduler']['warmup_steps'],
                    last_epoch=self.config['scheduler']['last_step'],
                )
                self._learning_rate_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self._optimizer,
                    schedulers=[warmup_lr_schr, cosine_lr_schr],
                    milestones=[self.config['scheduler']['warmup_steps']],
                    last_epoch=self.config['scheduler']['last_step'],
                )
            else:
                self._optimizer = None
                self._learning_rate_scheduler = None
            optimizer = self._optimizer
        return optimizer

    def update_learning_rate(self, stage: Literal['Step', 'Epoch'], **kwargs):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        if stage == 'Step':
            self._learning_rate_scheduler.step()
        return

    def mask_x(self, x: torch.Tensor, x_dict: dict[str, int], mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        label = x.clone()
        mask_probability = torch.full(x.shape, mask_ratio, dtype=torch.float, device=self.device_descriptor)
        mask_indices = torch.bernoulli(mask_probability).to(self.device_descriptor).bool()
        label[~mask_indices] = -1

        mask_mask_indices = torch.bernoulli(torch.full(x.shape, 0.8, dtype=torch.float, device=self.device_descriptor)).bool() & mask_indices
        x[mask_mask_indices] = x_dict['__MASK__']

        mask_optr_indices = torch.bernoulli(torch.full(x.shape, 0.5, dtype=torch.float, device=self.device_descriptor)).bool() & mask_indices & ~mask_mask_indices
        x[mask_optr_indices] = torch.randint(2, len(x_dict), x.shape, dtype=torch.long, device=self.device_descriptor)[mask_optr_indices]

        return x, label

    def train(self, minibatch: Data) -> tuple[torch.Tensor, OrderedDict]:
        minibatch = minibatch.to(self.device_descriptor)
        x, label = self.mask_x(minibatch.x, self.train_dataset.x_dict['o2i'], self.config['dataset']['mask_ratio'])
        output = self.model(x, minibatch.edge_index)

        loss = torch.nn.functional.cross_entropy(output, label.squeeze(1), ignore_index=-1)

        logs = OrderedDict({
            'loss': (loss, lambda x: f'{x:.4f}'),
        })
        return loss, logs

    def eval(self, minibatch: Data) -> tuple[torch.Tensor, torch.Tensor]:
        minibatch = minibatch.to(self.device_descriptor)
        if self.config['mode'] == 'Train':
            x_dict = self.train_dataset.x_dict['o2i']
        if self.config['mode'] == 'Test':
            x_dict = self.test_dataset.x_dict['o2i']
        x, label = self.mask_x(minibatch.x, x_dict, self.config['dataset']['mask_ratio'])
        output = torch.softmax(self.model(x, minibatch.edge_index), dim=-1)
        # Return Output & Golden
        return output, label

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict:
        all_outputs = torch.cat(all_outputs)
        all_goldens = torch.cat(all_goldens).squeeze()

        val_indices = all_goldens != -1
        all_outputs = all_outputs[val_indices]
        all_goldens = all_goldens[val_indices]

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

    def prepare_cli(self):
        self.meta = SSLDataset.load_meta(self.config['cli']['meta_filepath'], encode_type='operator')
        self.x_dict = SSLDataset.get_x_dict(self.meta, standard_onnx=True)
        self.logger.info(f'    -> Nodes Dict Size: {len(self.x_dict["o2i"])}')
        self.node_dict_size = len(self.x_dict["o2i"])

    def cli(self, device_descriptor):

        def cleanse_graph(graph: networkx.DiGraph):
            cleansed_graph = networkx.DiGraph()
            cleansed_graph.add_nodes_from(graph.nodes(data=True))
            cleansed_graph.add_edges_from(graph.edges(data=True))
            for node_index in cleansed_graph.nodes():
                cleansed_graph.nodes[node_index]['operator'] = cleansed_graph.nodes[node_index]['features']['operator']
            return cleansed_graph

        operator_embeddings = self.model.encoder.node_embedding_layer.weight.detach().to('cpu').numpy().tolist()
        assert len(operator_embeddings) == len(self.x_dict['o2i'])

        opemb_dict = dict()
        dagemb_dict = dict()
        s2p_hash_dict = dict() # son -> parent
        graphs = dict()
        if self.config['cli']['input_type'] == 'instance':
            for instance in Dataset.load_instances(self.config['cli']['instances_dirpath']):
                graph = cleanse_graph(instance.network.graph)
                graph_hash = Network.hash(graph, node_attr='operator')
                if graph.number_of_nodes() <= self.config['cli']['node_size_limit']:
                    continue
                graphs[graph_hash] = graph
                s2p_hash_dict[graph_hash] = graph_hash

        if self.config['cli']['input_type'] == 'subgraph':
            for subgraph_filepath in pathlib.Path(self.config['cli']['subgraphs_dirpath']).iterdir():
                subgraph_hash, subgraph, _ = load_pickle(subgraph_filepath)
                graphs[subgraph_hash] = subgraph
                s2p_hash_dict[subgraph_hash] = subgraph.graph['graph_hash']

        op_details_dict = dict()
        for graph_hash, graph in graphs.items():

            op_detail = dict()
            for node_index in graph.nodes():
                node_identifier = Network.get_node_identifier_from_features(graph.nodes[node_index], mode='type')
                op_detail[node_identifier] = op_detail.get(node_identifier, 0) + 1
            op_details_dict[graph_hash] = op_detail

            data = SSLDataset.get_block_data((graph_hash, graph, None), self.x_dict, 'operator').to(device_descriptor)
            for index in torch.unique(data.x):
                operator_id = self.x_dict['i2o'][index]
                if operator_id not in opemb_dict:
                    opemb_dict[operator_id] = operator_embeddings[index]
            dagemb_dict[graph_hash] = torch.mean(self.model.encoder(data.x, data.edge_index), dim=0, keepdim=False).detach().to('cpu').numpy().tolist()

        result_dict = dict(
            opembs = opemb_dict,
            dagembs = dagemb_dict,
            s2p_hash = s2p_hash_dict,
            op_details = op_details_dict
        )
        result_filepath = pathlib.Path(self.config['cli']['result_filepath'])
        save_pickle(result_dict, result_filepath)