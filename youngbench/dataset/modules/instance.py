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


import pathlib
import semantic_version

from typing import Set, List, Dict

from youngbench.dataset.modules.meta import Meta
from youngbench.dataset.modules.network import Network
from youngbench.dataset.modules.stamp import Stamp
from youngbench.dataset.modules.model import Model

from youngbench.dataset.utils.io import hash_strings, read_json, write_json
from youngbench.dataset.utils.extraction import extract_network


class Instance(object):
    def __init__(self,
        network: Network = Network(),
        models: List[Model] = list(),
        version: semantic_version.Version = semantic_version.Version('0.0.0')
    ) -> None:
        self._meta_filename = 'meta.json'
        self._meta = Meta()

        self._network_filename = 'network.json'
        self._network = Network()

        self._stamps_filename = 'stamps.json'
        self._stamps = set()

        self._uniques_filename = 'uniques.json'
        self._uniques = list()

        self._models_dirname = 'models'
        self._models = dict()

        self._mode = None

        self._legacy = False

        self.setup_network(network)
        self.insert_models(models)
        self.release(version)

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def network(self) -> Network:
        return self._network

    @property
    def stamps(self) -> Set[Stamp]:
        return self._stamps

    @property
    def uniques(self) -> List[str]:
        return self._uniques

    @property
    def models(self) -> Dict[str, Model]:
        return self._models

    @property
    def mode(self):
        return self._mode

    @property
    def mode_open(self) -> int:
        return 0B1

    @property
    def mode_close(self) -> int:
        return 0B0

    @property
    def latest_version(self) -> semantic_version.Version:
        latest_version = semantic_version.Version('0.0.0')
        for stamp in self.stamps:
            latest_version = max(latest_version, stamp.version)
        return latest_version

    @property
    def checksum(self) -> str:
        ids = list()
        for model in self.models.values():
            ids.append(model.meta.identifier)
            ids.append(model.identifier)
        return hash_strings(ids)

    @property
    def identifier(self) -> str:
        return self.network.identifier

    def __len__(self) -> int:
        return len(self.models)

    @property
    def status(self) -> int:
        status = ((len(self.network) == 0) << 2) + (self.meta.release << 1) + (self.meta.retired or self._legacy)
        # 0B100 -> It has never been added before.
        # 0BX1X
        # 0BX10 -> Release (New)
        # 0BX11 -> Retired (Old)
        # 0BX0X
        # 0BX00 -> New
        # 0BX01 -> Old
        assert 0B000 <= status and status <= 0B111, f'Invalid status code: {status}.'
        assert (status >> 1) != 0B11, f'Invalid status code: {status}.'
        assert status != 0B101, f'Invalid status code: {status}.'
        return status

    @property
    def is_fresh(self) -> bool:
        return self.status == 0B100

    @property
    def is_internal(self) -> bool:
        return (self.status & 0B010) == 0B010
    @property
    def is_external(self) -> bool:
        return (self.status & 0B010) == 0B000

    @property
    def is_new(self) -> bool:
        return (self.status & 0B001) == 0B000

    @property
    def is_old(self) -> bool:
        return (self.status & 0B001) == 0B001

    def set_new(self) -> None:
        self._legacy = False
        return

    def set_old(self) -> None:
        self._legacy = True
        return

    def load(self, instance_dirpath: pathlib.Path) -> None:
        assert instance_dirpath.is_dir(), f'There is no \"Instance\" can be loaded from the specified directory \"{instance_dirpath.absolute()}\".'
        meta_filepath = instance_dirpath.joinpath(self._meta_filename)
        self._load_meta(meta_filepath)
        network_filepath = instance_dirpath.joinpath(self._network_filename)
        self._load_network(network_filepath)

        stamps_filepath = instance_dirpath.joinpath(self._stamps_filename)
        self._load_stamps(stamps_filepath)
        uniques_filepath = instance_dirpath.joinpath(self._uniques_filename)
        self._load_uniques(uniques_filepath)
        models_dirpath = instance_dirpath.joinpath(self._models_dirname)
        self._load_models(models_dirpath)
        return 

    def save(self, instance_dirpath: pathlib.Path) -> None:
        assert not instance_dirpath.is_dir(), f'\"Instance\" can not be saved into the specified directory \"{instance_dirpath.absolute()}\".'
        network_filepath = instance_dirpath.joinpath(self._network_filename)
        self._save_network(network_filepath)
        meta_filepath = instance_dirpath.joinpath(self._meta_filename)
        self._save_meta(meta_filepath)

        models_dirpath = instance_dirpath.joinpath(self._models_dirname)
        self._save_models(models_dirpath)
        uniques_filepath = instance_dirpath.joinpath(self._uniques_filename)
        self._save_uniques(uniques_filepath)
        stamps_filepath = instance_dirpath.joinpath(self._stamps_filename)
        self._save_stamps(stamps_filepath)
        return

    def _load_meta(self, meta_filepath: pathlib.Path) -> None:
        assert meta_filepath.is_file(), f'There is no \"Meta\" can be loaded from the specified path \"{meta_filepath.absolute()}\".'
        meta = read_json(meta_filepath)
        self._meta = Meta(**meta)
        return

    def _save_meta(self, meta_filepath: pathlib.Path) -> None:
        assert not meta_filepath.is_file(), f'\"Meta\" can not be saved into the specified path \"{meta_filepath.absolute()}\".'
        meta = self._meta.dict
        write_json(meta, meta_filepath)
        return

    def _load_network(self, network_filepath: pathlib.Path) -> None:
        assert network_filepath.is_file(), f'There is no \"Prototype\" can be loaded from the specified path \"{network_filepath.absolute()}\".'
        self._network.load(network_filepath)
        return

    def _save_network(self, network_filepath: pathlib.Path) -> None:
        assert not network_filepath.is_file(), f'\"Prototype\" can not be saved into the specified path \"{network_filepath.absolute()}\".'
        self._network.save(network_filepath)
        return

    def _load_stamps(self, stamps_filepath: pathlib.Path) -> None:
        assert stamps_filepath.is_file(), f'There is no \"Stamp\"s can be loaded from the specified path \"{stamps_filepath.absolute()}\".'
        stamps = read_json(stamps_filepath)
        self._stamps = set()
        for stamp in stamps:
            self.stamps.add(Stamp(**stamp))
        return

    def _save_stamps(self, stamps_filepath: pathlib.Path) -> None:
        assert not stamps_filepath.is_file(), f'\"Stamp\"s can not be saved into the specified path \"{stamps_filepath.absolute()}\".'
        stamps = list()
        for stamp in self.stamps:
            stamps.append(stamp.dict)
        write_json(stamps, stamps_filepath)
        return

    def _load_uniques(self, uniques_filepath: pathlib.Path) -> None:
        assert uniques_filepath.is_file(), f'There is no \"Unique\"s can be loaded from the specified path \"{uniques_filepath.absolute()}\".'
        self._uniques = read_json(uniques_filepath)
        assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
        return

    def _save_uniques(self, uniques_filepath: pathlib.Path) -> None:
        assert not uniques_filepath.is_file(), f'\"Unique\"s can not be saved into the specified path \"{uniques_filepath.absolute()}\".'
        assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
        write_json(self._uniques, uniques_filepath)
        return

    def _load_models(self, models_dirpath: pathlib.Path) -> None:
        assert models_dirpath.is_dir(), f'There is no \"Model\" can be loaded from the specified directory \"{models_dirpath.absolute()}\".'
        for index, identifier in enumerate(self._uniques):
            model_dirpath = models_dirpath.joinpath(f'{index}-{identifier}')
            self._models[identifier] = Instance()
            self._models[identifier].load(model_dirpath)
        return

    def _save_models(self, models_dirpath: pathlib.Path) -> None:
        assert not models_dirpath.is_dir(), f'\"Model\"s can not be saved into the specified directory \"{models_dirpath.absolute()}\".'
        for index, identifier in enumerate(self._uniques):
            model_dirpath = models_dirpath.joinpath(f'{index}-{identifier}')
            model = self._models[identifier]
            model.save(model_dirpath)
        return

    def acquire(self, version: semantic_version.Version) -> 'Instance':
        if (self.meta.release and self.meta.release_version <= version) and (not self.meta.retired or version < self.meta.retired_version):
            instance = Instance(network=self.network, models=[model for model in self.models.values() if model.acquire(version) is not None], version=version)
        else:
            instance = None
        return instance

    def check(self) -> None:
        # Check Models
        assert len(self.uniques) == len(self.models), f'The number of \"Model\"s does not match the number of \"Unique\"s.'
        for identifier, model in zip(self.uniques, self.models.values()):
            assert identifier == model.identifier, f'The \"Identifier={model.identifier}\" of \"Model\" does not match \"Unique={identifier}\" '
            model.check()
        # Check Stamps
        for stamp in self.stamps:
            instance = self.acquire(stamp.version)
            assert stamp.checksum == instance.checksum, f'The \"Checksum={instance.checksum}\" of \"Instance\" (Version={stamp.version}) does not match \"Stamp={stamp.checksum}\"'
        return

    def setup_network(self, network: Network) -> None:
        assert isinstance(network, Network), f'Parameter network must be a \"Network\" object instead \"{type(network)}\"!'
        if self.is_fresh:
            self._network = network
            self.set_new()
        return

    def clear_network(self, network: Network) -> None:
        assert isinstance(network, Network), f'Parameter network must be a \"Network\" object instead \"{type(network)}\"!'
        all_old = True
        for model in self.models.values():
            all_old = all_old and model.meta.retired

        if all_old:
            self.set_old()
        return

    def insert(self, model: Model) -> None:
        if self.is_fresh:
            return
        if self.is_new:
            if self.network == extract_network(model):
                new_model = model.copy()
                self.models[new_model.identifier] = new_model
        return 

    def delete(self, model: Model) -> None:
        if self.is_fresh:
            return
        if self.is_new:
            if self.network == extract_network(model):
                old_model = self.models.get(model.identifier, Model())
                old_model.set_old()
        return

    def insert_models(self, models: List[Model]) -> None:
        for model in models:
            self.insert(model)
        return

    def delete_models(self, models: List[Model]) -> None:
        for model in models:
            self.insert(model)
        return

    def release(self, version: semantic_version.Version) -> None:
        if self.is_fresh or version == semantic_version.Version('0.0.0'):
            return

        assert self.latest_version < version, (
            f'Version provided less than or equal to the latest version:\n'
            f'Provided: {version}'
            f'Latest: {self.latest_version}'
        )

        for identifier, model in self._models.items():
            if model.is_external:
                if model.is_new:
                    self._uniques.append(identifier)
                if model.is_old:
                    self._models.pop(identifier)
            model.release(version)

        stamp = Stamp(
            str(version),
            self.checksum,
        )
        if stamp in self._stamps:
            return
        else:
            self._stamps.add(stamp)

        if self.meta.release:
            if self.is_old:
                self.meta.set_retired(version)
        else:
            if self.is_new:
                self.meta.set_release(version)

        return