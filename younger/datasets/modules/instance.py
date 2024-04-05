#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-11 11:19
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import onnx
import pathlib

from younger.commons.version import semantic_release, str_to_sem

from younger.datasets.modules.meta import Meta
from younger.datasets.modules.network import Network

from younger.datasets.utils.io import load_json, save_json, load_model, check_model
from younger.datasets.utils.constants import InstanceLabelName
from younger.datasets.utils.translation import trans_model_proto


class Instance(object):
    def __init__(
            self,
            model: onnx.ModelProto | pathlib.Path | None = None,
            labels: dict[str, str] | None = None,
            version: semantic_release.Version | None = None,
    ) -> None:
        model = model or onnx.ModelProto()
        labels = labels or dict()
        version = version or str_to_sem('0.0.0')

        self._meta_filename = 'meta.json'
        self._meta: Meta = Meta(fresh_checker=self.fresh_checker)

        self._network_dirname = 'network'
        self._network = Network()

        self._labels_filename = 'label.json'
        self._labels = dict()
        for attribute in InstanceLabelName.attributes:
            label_name = getattr(InstanceLabelName, attribute)
            self._labels[label_name] = None

        self._unique_filename = 'unique.id'
        self._unique = None

        self.tag(labels)
        self.setup(model)
        self.release(version)

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def labels(self) -> dict[str, str]:
        return self._labels

    @property
    def network(self) -> Network:
        return self._network

    @property
    def unique(self) -> str | None:
        return self._unique

    def fresh_checker(self) -> bool:
        return self._unique is None

    def load(self, instance_dirpath: pathlib.Path) -> None:
        assert instance_dirpath.is_dir(), f'There is no \"Instance\" can be loaded from the specified directory \"{instance_dirpath.absolute()}\".'
        meta_filepath = instance_dirpath.joinpath(self._meta_filename)
        self._load_meta(meta_filepath)
        unique_filepath = instance_dirpath.joinpath(self._unique_filename)
        self._load_unique(unique_filepath)
        labels_filepath = instance_dirpath.joinpath(self._labels_filename)
        self._load_labels(labels_filepath)
        network_dirpath = instance_dirpath.joinpath(self._network_dirname)
        self._load_network(network_dirpath)
        return 

    def save(self, instance_dirpath: pathlib.Path) -> None:
        assert not instance_dirpath.is_dir(), f'\"Instance\" can not be saved into the specified directory \"{instance_dirpath.absolute()}\".'
        meta_filepath = instance_dirpath.joinpath(self._meta_filename)
        self._save_meta(meta_filepath)
        unique_filepath = instance_dirpath.joinpath(self._unique_filename)
        self._save_unique(unique_filepath)
        labels_filepath = instance_dirpath.joinpath(self._labels_filename)
        self._save_labels(labels_filepath)
        network_dirpath = instance_dirpath.joinpath(self._network_dirname)
        self._save_network(network_dirpath)
        return

    def _load_meta(self, meta_filepath: pathlib.Path) -> None:
        assert meta_filepath.is_file(), f'There is no \"Meta\" can be loaded from the specified path \"{meta_filepath.absolute()}\".'
        self._meta.load(meta_filepath)
        return

    def _save_meta(self, meta_filepath: pathlib.Path) -> None:
        assert not meta_filepath.is_file(), f'\"Meta\" can not be saved into the specified path \"{meta_filepath.absolute()}\".'
        self._meta.save(meta_filepath)
        return

    def _load_unique(self, unique_filepath: pathlib.Path) -> None:
        assert unique_filepath.is_file(), f'There is no \"Unique ID\" can be loaded from the specified path \"{unique_filepath.absolute()}\".'
        self._unique = load_json(unique_filepath)
        return

    def _save_unique(self, unique_filepath: pathlib.Path) -> None:
        assert not unique_filepath.is_file(), f'\"Unique ID\" can not be saved into the specified path \"{unique_filepath.absolute()}\".'
        save_json(self._unique, unique_filepath)
        return

    def _load_labels(self, labels_filepath: pathlib.Path) -> None:
        assert labels_filepath.is_file(), f'There is no \"Lables\" can be loaded from the specified path \"{labels_filepath.absolute()}\".'
        self._labels = load_json(labels_filepath)
        return

    def _save_labels(self, labels_filepath: pathlib.Path) -> None:
        assert not labels_filepath.is_file(), f'\"Labels\" can not be saved into the specified path \"{labels_filepath.absolute()}\".'
        save_json(self._labels, labels_filepath)
        return

    def _load_network(self, network_dirpath: pathlib.Path) -> None:
        assert network_dirpath.is_dir(), f'There is no \"Network\" can be loaded from the specified directory \"{network_dirpath.absolute()}\".'
        self._network.load(network_dirpath)
        return

    def _save_network(self, network_dirpath: pathlib.Path) -> None:
        assert not network_dirpath.is_dir(), f'\"Network\" can not be saved into the specified directory \"{network_dirpath.absolute()}\".'
        self._network.save(network_dirpath)
        return

    def tag(self, labels: dict[str, str]) -> None:
        if self.meta.is_fresh:
            for label_key, label_value in labels.items():
                self._labels[label_key] = label_value
        return

    def setup(self, model_handler: onnx.ModelProto | pathlib.Path) -> None:
        assert isinstance(model_handler, onnx.ModelProto) or isinstance(model_handler, pathlib.Path), f'Argument \"model_handler\" must be an ONNX Model Proto (onnx.ModelProto) or a Path (pathlib.Path) instead \"{type(model_handler)}\"!'
        # Unset Unique Due To OOM and Huge External Data
        if self.meta.is_fresh:
            unique = None

            if check_model(model_handler):
                if isinstance(model_handler, onnx.ModelProto):
                    model = model_handler
                if isinstance(model_handler, pathlib.Path):
                    model = load_model(model_handler)

                self._network = Network(trans_model_proto(model, neglect_tensor_values=True))
            # TODO: Set Unique
            self._unique = unique
        return

    def copy(self) -> 'Instance':
        # TODO:  Optimize this method, as assigning network values involves significant memory consumption.
        instance = Instance()
        instance._network = self._network
        instance._unique = self._unique
        return instance

    def insert(self) -> bool:
        if self.meta.is_fresh:
            return False
        else:
            self.meta.set_new()
            return True

    def delete(self) -> bool:
        if self.meta.is_fresh:
            return False
        else:
            self.meta.set_old()
            return True

    def release(self, version: semantic_release.Version) -> None:
        if self.meta.is_fresh or version == str_to_sem('0.0.0'):
            return

        if self.meta.release:
            if self.meta.is_old:
                self.meta.set_retired(version)
        else:
            if self.meta.is_new:
                self.meta.set_release(version)
        return
