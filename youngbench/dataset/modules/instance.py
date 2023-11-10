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

from typing import Set, List, Dict, Optional

from youngbench.dataset.modules.meta import Meta
from youngbench.dataset.modules.network import Prototype, Network
from youngbench.dataset.modules.stamp import Stamp

from youngbench.dataset.utils.io import hash_strings, read_json, write_json
from youngbench.logging import logger


class Instance(object):
    def __init__(
            self,
            prototype: Optional[Prototype] = None,
            networks: Optional[List[Network]] = None,
            version: semantic_version.Version = semantic_version.Version('0.0.0')
    ) -> None:
        prototype = prototype or Prototype()
        networks = networks or list()

        self._meta_filename = 'meta.json'
        self._meta = Meta()

        self._prototype_filename = 'prototype.json'
        self._prototype = Prototype()

        self._stamps_filename = 'stamps.json'
        self._stamps = set()

        self._uniques_filename = 'uniques.json'
        self._uniques = list()

        self._networks_dirname = 'networks'
        self._networks = dict() 

        self._mode = None

        self._legacy = False

        self.setup_prototype(prototype)
        self.insert_networks(networks)
        self.release(version)

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def prototype(self) -> Prototype:
        return self._prototype

    @property
    def stamps(self) -> Set[Stamp]:
        return self._stamps

    @property
    def uniques(self) -> List[str]:
        return self._uniques

    @property
    def networks(self) -> Dict[str, Network]:
        return self._networks

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
        ids.append(self.identifier)
        for identifier in self.uniques:
            network = self.networks[identifier]
            if network.is_release:
                ids.append(network.identifier)
        return hash_strings(ids)

    @property
    def identifier(self) -> str:
        return self.prototype.identifier

    def __len__(self) -> int:
        return len(self.networks)

    @property
    def status(self) -> int:
        status = ((len(self.prototype) == 0) << 2) + (self.meta.release << 1) + (self.meta.retired or self._legacy)
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
    def is_release(self) -> bool:
        return (self.status & 0B011) == 0B010

    @property
    def is_retired(self) -> bool:
        return (self.status & 0B011) == 0B011

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
        prototype_filepath = instance_dirpath.joinpath(self._prototype_filename)
        self._load_prototype(prototype_filepath)

        stamps_filepath = instance_dirpath.joinpath(self._stamps_filename)
        self._load_stamps(stamps_filepath)
        uniques_filepath = instance_dirpath.joinpath(self._uniques_filename)
        self._load_uniques(uniques_filepath)
        networks_dirpath = instance_dirpath.joinpath(self._networks_dirname)
        self._load_networks(networks_dirpath)
        return 

    def save(self, instance_dirpath: pathlib.Path) -> None:
        assert not instance_dirpath.is_dir(), f'\"Instance\" can not be saved into the specified directory \"{instance_dirpath.absolute()}\".'
        prototype_filepath = instance_dirpath.joinpath(self._prototype_filename)
        self._save_prototype(prototype_filepath)
        meta_filepath = instance_dirpath.joinpath(self._meta_filename)
        self._save_meta(meta_filepath)

        networks_dirpath = instance_dirpath.joinpath(self._networks_dirname)
        self._save_networks(networks_dirpath)
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

    def _load_prototype(self, prototype_filepath: pathlib.Path) -> None:
        assert prototype_filepath.is_file(), f'There is no \"Prototype\" can be loaded from the specified path \"{prototype_filepath.absolute()}\".'
        self._prototype.load(prototype_filepath)
        return

    def _save_prototype(self, prototype_filepath: pathlib.Path) -> None:
        assert not prototype_filepath.is_file(), f'\"Prototype\" can not be saved into the specified path \"{prototype_filepath.absolute()}\".'
        self._prototype.save(prototype_filepath)
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

    def _load_networks(self, networks_dirpath: pathlib.Path) -> None:
        if len(self._uniques) == 0:
            return
        assert networks_dirpath.is_dir(), f'There is no \"Network\" can be loaded from the specified directory \"{networks_dirpath.absolute()}\".'
        for index, identifier in enumerate(self._uniques):
            logger.info(f' = [YBD] = \u2514 No.{index} Network: {identifier}')
            network_dirpath = networks_dirpath.joinpath(f'{index}-{identifier}')
            self._networks[identifier] = Network()
            self._networks[identifier].load(network_dirpath)
        return

    def _save_networks(self, networks_dirpath: pathlib.Path) -> None:
        if len(self._uniques) == 0:
            return
        assert not networks_dirpath.is_dir(), f'\"Network\"s can not be saved into the specified directory \"{networks_dirpath.absolute()}\".'
        for index, identifier in enumerate(self._uniques):
            logger.info(f' = [YBD] = \u2514 No.{index} Network: {identifier}')
            network_dirpath = networks_dirpath.joinpath(f'{index}-{identifier}')
            network = self._networks[identifier]
            network.save(network_dirpath)
        return

    def acquire(self, version: semantic_version.Version) -> 'Instance':
        if (self.meta.release and self.meta.release_version <= version) and (not self.meta.retired or version < self.meta.retired_version):
            instance = Instance()
            instance.setup_prototype(self.prototype)
            for index, identifier in enumerate(self._uniques):
                network = self._networks[identifier].acquire(version)
                if network is not None:
                    logger.info(f' = [YBD] = Acquired \u250c No.{index} Network: {identifier}')
                    instance._networks[identifier] = network
        else:
            instance = None
        return instance

    def check(self) -> None:
        assert len(self.uniques) == len(self.networks), f'The number of \"Network\"s does not match the number of \"Unique\"s.'
        for identifier in self.uniques:
            network = self.networks[identifier]
            assert identifier == network.identifier, f'The \"Identifier={network.identifier}\" of \"Network\" does not match \"Unique={identifier}\" '
            network.check()
        return

    def setup_prototype(self, prototype: Prototype) -> bool:
        assert isinstance(prototype, Prototype), f'Parameter prototype must be a \"Prototype\" object instead \"{type(prototype)}\"!'
        if self.is_fresh:
            self._prototype = prototype
            self.set_new()
            return True
        return False

    def clear_prototype(self, prototype: Prototype) -> bool:
        assert isinstance(prototype, Prototype), f'Parameter prototype must be a \"Prototype\" object instead \"{type(prototype)}\"!'
        all_old = True
        for network in self.networks.values():
            all_old = all_old and network.meta.retired

        if all_old:
            self.set_old()
            return True
        return False

    def insert(self, network: Network) -> bool:
        if self.is_fresh:
            return False
        if self.is_new and self.prototype == network.prototype:
            new_network = self._networks.get(network.identifier, None)
            if new_network is None:
                new_network = network.copy()
            flags_sum = new_network.insert_models(network.models.values())
            self._networks[new_network.identifier] = new_network
            return flags_sum > 0
        return False

    def delete(self, network: Network) -> bool:
        if self.is_fresh:
            return False
        if self.is_new and self.prototype == network.prototype:
            old_network = self._networks.get(network.identifier, None)
            if old_network is None:
                return False
            flags_sum = old_network.delete_models(network.models.values())
            old_network.set_old()
            return flags_sum > 0
        return False

    def insert_networks(self, networks: List[Network]) -> int:
        flags = list()
        for network in networks:
            flags.append(self.insert(network))
        return sum(flags)

    def delete_networks(self, networks: List[Network]) -> int:
        flags = list()
        for network in networks:
            flags.append(self.insert(network))
        return sum(flags)

    def release(self, version: semantic_version.Version) -> None:
        if self.is_fresh or version == semantic_version.Version('0.0.0'):
            return

        assert self.latest_version < version, (
            f'Version provided less than or equal to the latest version:\n'
            f'Provided: {version}\n'
            f'Latest: {self.latest_version}'
        )

        for identifier, network in self._networks.items():
            if network.is_external:
                if network.is_new:
                    self._uniques.append(identifier)
                if network.is_old:
                    self._networks.pop(identifier)
            network.release(version)

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