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
import networkx
import semantic_version

from typing import Set, List, Dict

from youngbench.dataset.modules.meta import Meta
from youngbench.dataset.modules.stamp import Stamp
from youngbench.dataset.modules.instance import Instance
from youngbench.dataset.modules.prototype import Prototype

from youngbench.dataset.utils import hash_strings, read_json, write_json
from youngbench.dataset.logging import logger


class Network(object):
    mode_codes = dict(
        close = 0B0,
        open = 0B1,
    )

    def __init__(self, prototype: networkx.DiGraph = None, instances: List[Instance] = None, version: semantic_version.Version = None) -> None:
        self._meta_filename = 'meta.json'
        self._meta = Meta()

        self._prototype_filename = 'prototype.json'
        self._prototype = Prototype()

        self._stamps_filename = 'stamps.json'
        self._stamps = set()

        self._uniques_filename = 'uniques.json'
        self._uniques = list()

        self._instances_dirname = 'instances'
        self._instances = list()

        self._latest_version = semantic_version.Version('0.0.0')

        self._mode_code = 0B0
        self._map = dict()

        self._legacy = False

        if prototype:
            self.open()
            self.set_prototype(prototype)
            self.add_instances(instances)
            self.release(version)
            self.close()

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
    def instances(self) -> List[Instance]:
        return self._instances

    @property
    def latest_version(self) -> semantic_version.Version:
        return self._latest_version

    @property
    def checksum(self) -> str:
        if len(self.map) != 0:
            logger.warn(f'There are still unreleased \"Instance\"s; the checksum may be incorrect.')
        ids = list()
        for instance in self.instances:
            ids.append(instance.identifier)
            ids.append(instance.meta.identifier)

        return hash_strings(ids)

    @property
    def identifier(self) -> str:
        return self.prototype.identifier

    @property
    def mode_code(self) -> bool:
        return self._mode_code

    @property
    def map(self) -> Dict[str, Instance]:
        return self._map

    def get_mode_code(self, mode_pattern: str) -> int:
        return self.__class__.mode_codes[mode_pattern]

    def set_mode_code(self, mode_pattern: str) -> None:
        self._mode_code = self.__class__.mode_codes[mode_pattern]

        return

    def open(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'The mode of \"Network\" is \"open\", do not need open it again.')
        self.set_mode_code('open')
        self._map = dict()
        for identifier, instance in zip(self.uniques, self.instances):
            instance.open()
            self._map[identifier] = instance
        return 

    def __enter__(self) -> None:
        self.open()
        return self

    def close(self) -> None:
        if self.mode_code == self.get_mode_code('close'):
            logger.warn(f'The mode of \"Network\" is \"close\", do not need close it again.')
        self.set_mode_code('close')
        if len(self._map) != 0:
            logger.warn(f'There are still unreleased \"Instance\"s; all are lost.')
        for identifier, instance in zip(self.uniques, self.instances):
            instance.close()
        self._map = dict()

        return

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

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
    def is_new(self) -> bool:
        return (self.status & 0B001) == 0B0

    def set_new(self) -> None:
        self._legacy = False

        return

    @property
    def is_old(self) -> bool:
        return (self.status & 0B001) == 0B1

    def set_old(self) -> None:
        self._legacy = True

        return

    def load(self, network_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert network_dirpath.is_dir(), f'There is no \"Network\" can be loaded from the specified directory \"{network_dirpath.absolute()}\".'
            meta_filepath = network_dirpath.joinpath(self._meta_filename)
            self._load_meta(meta_filepath)
            prototype_filepath = network_dirpath.joinpath(self._prototype_filename)
            self._load_prototype(prototype_filepath)

            stamps_filepath = network_dirpath.joinpath(self._stamps_filename)
            self._load_stamps(stamps_filepath)
            uniques_filepath = network_dirpath.joinpath(self._uniques_filename)
            self._load_uniques(uniques_filepath)
            instances_dirpath = network_dirpath.joinpath(self._instances_dirname)
            self._load_instances(instances_dirpath)

        return 

    def save(self, network_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not network_dirpath.is_dir(), f'\"Network\" can not be saved into the specified directory \"{network_dirpath.absolute()}\".'
            network_dirpath.mkdir()
            prototype_filepath = network_dirpath.joinpath(self._prototype_filename)
            self._save_prototype(prototype_filepath)
            meta_filepath = network_dirpath.joinpath(self._meta_filename)
            self._save_meta(meta_filepath)

            instances_dirpath = network_dirpath.joinpath(self._instances_dirname)
            self._save_instances(instances_dirpath)
            uniques_filepath = network_dirpath.joinpath(self._uniques_filename)
            self._save_uniques(uniques_filepath)
            stamps_filepath = network_dirpath.joinpath(self._stamps_filename)
            self._save_stamps(stamps_filepath)

        return 

    def _load_meta(self, meta_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert meta_filepath.is_file(), f'There is no \"Meta\" can be loaded from the specified path \"{meta_filepath.absolute()}\".'
            meta = read_json(meta_filepath)
            self._meta = Meta(**meta)

        return

    def _save_meta(self, meta_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not meta_filepath.is_file(), f'\"Meta\" can not be saved into the specified path \"{meta_filepath.absolute()}\".'
            write_json(self.meta.dict, meta_filepath)

        return

    def _load_prototype(self, prototype_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert prototype_filepath.is_file(), f'There is no \"Prototype\" can be loaded from the specified path \"{prototype_filepath.absolute()}\".'
            self._prototype.load(prototype_filepath)

        return

    def _save_prototype(self, prototype_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not prototype_filepath.is_file(), f'\"Prototype\" can not be saved into the specified path \"{prototype_filepath.absolute()}\".'
            self._prototype.save(prototype_filepath)

        return

    def _load_stamps(self, stamps_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert stamps_filepath.is_file(), f'There is no \"Stamp\"s can be loaded from the specified path \"{stamps_filepath.absolute()}\".'

            stamps_and_latest_version = read_json(stamps_filepath)
            stamps = stamps_and_latest_version['stamps']
            for stamp in stamps:
                stamp = Stamp(**stamp)
                self._stamps.add(stamp)

            latest_version = stamps_and_latest_version['latest_version']
            self._latest_version = semantic_version.Version(latest_version)

        return

    def _save_stamps(self, stamps_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not stamps_filepath.is_file(), f'\"Stamp\"s can not be saved into the specified path \"{stamps_filepath.absolute()}\".'

            stamps_filepath.touch()
            stamps_and_latest_version = dict()
            stamps = list()
            for stamp in self._stamps:
                stamps.append(stamp.dict)
            stamps_and_latest_version['stamps'] = stamps

            latest_version = str(self._latest_version)
            stamps_and_latest_version['latest_version'] = latest_version

            write_json(stamps_and_latest_version, stamps_filepath)

        return

    def _load_uniques(self, uniques_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert uniques_filepath.is_file(), f'There is no \"Unique\"s can be loaded from the specified path \"{uniques_filepath.absolute()}\".'

            self._uniques = read_json(uniques_filepath)
            assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'

        return

    def _save_uniques(self, uniques_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not uniques_filepath.is_file(), f'\"Unique\"s can not be saved into the specified path \"{uniques_filepath.absolute()}\".'

            uniques_filepath.touch()
            assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
            write_json(self._uniques, uniques_filepath)

        return

    def _load_instances(self, instances_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert instances_dirpath.is_dir(), f'There is no \"Instance\" can be loaded from the specified directory \"{instance_dirpath.absolute()}\".'

            for index, identifier in enumerate(self._uniques):
                instance_dirpath = instances_dirpath.joinpath(f'{index}-{identifier}')
                instance = Instance()
                instance.load(instance_dirpath)
                self._instances.append(instance)

        return

    def _save_instances(self, instances_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not instances_dirpath.is_dir(), f'\"Instances\"s can not be saved into the specified directory \"{instances_dirpath.absolute()}\".'

            instances_dirpath.mkdir()
            for index, identifier in enumerate(self._uniques):
                instance_dirpath = instances_dirpath.joinpath(f'{index}-{identifier}')
                instance = self._instances[index]
                instance.save(instance_dirpath)

        return

    def acquire(self, version: semantic_version.Version) -> 'Network':
        assert self.mode_code == self.get_mode_code('close'), f'Can not acquire! The mode of \"Network\" is \"open\".'

        network = Network()
        network.open()
        for instance in self.instances:
            if instance.meta.release:
                if instance.meta.retired:
                    if instance.meta.release_version <= version and version < instance.meta.retired_version:
                        network.set_prototype(instance.prototype)
                        network.add()
                        network.add_instances([instance,])
                else:
                    if instance.meta.release_version <= version:
                        network.set_prototype(instance.prototype)
                        network.add()
                        network.add_instances([instance,])

        network.release(version)
        network.close()
        return network

    def check(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'No Check! The mode of \"Network\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            logger.info(f'Begin Check!')
            latest_version = semantic_version.Version('0.0.0')
            for stamp in self.stamps:
                latest_version = max(latest_version, stamp.version)
            assert self.latest_version == latest_version, f'\"Latest version\" incorrect, it should be \"{latest_version}\" instead \"{self.latest_version}\"'

            assert len(self.uniques) == len(self.instances), f'The number of \"Instances\"s does not match the number of \"Unique\"s.'

            for identifier, instance in zip(self.uniques, self.instances):
                assert identifier == instance.identifier, f'The \"Identifier={instance.identifier}\" of \"Instance\" does not match \"Unique={identifier}\" '

            for stamp in self.stamps:
                network = self.acquire(stamp.version)
                assert stamp.checksum == network.checksum, f'The \"Checksum={network.checksum}\" of \"Network\" (Version={stamp.version}) does not match \"Stamp={stamp.checksum}\"'

        return

    def set_prototype(self, prototype: Prototype) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                assert isinstance(prototype, Prototype), f'Parameter prototype must be a \"Prototype\" object instead \"{type(prototype)}\"!'
                self._prototype = prototype
            else:
                logger.info(f'\"Prototype\" has already been set, \"set\" operation will not take effect.')
        else:
            logger.warn(f'The mode of \"Network\" is \"close\", no action')

        return

    def add(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                logger.warn(f'Please set \"prototype\" first by using method: \"set_prototype(prototype)\".')
            if self.is_old:
                self.set_new()
        else:
            logger.warn(f'The mode of \"Network\" is \"close\", no action')

        return

    def delete(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                logger.warn(f'Please set \"prototype\" first by using method: \"set_prototype(prototype)\".')
            if self.is_new:
                self.set_old()
        else:
            logger.warn(f'The mode of \"Network\" is \"close\", no action')

        return

    def add_instances(self, instances: List[Instance]) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                logger.info(f'Please set \"prototype\" first by using method: \"set_prototype(prototype)\".')
            if self.is_new:
                for index, instance in enumerate(instances):
                    if self.prototype == instance.prototype:
                        instance_identifier = instance.identifier
                        if instance_identifier in self._map:
                            logger.info(f'Skip, \"Instance\" exists: {instance_identifier}.')
                        else:
                            self._map[instance_identifier] = Instance()
                            self._map[instance_identifier].open()
                            self._map[instance_identifier].set_model(instance.model)
                        self._map[instance_identifier].add()
                    else:
                        logger.info(f'Skip No.{index}, \"Prototype\" of the \"Instance\" does not match the one of the \"Network\".')
            else:
                logger.info(f'The \"Network\" is in \"Old\" status, please re-New it.')
        else:
            logger.warn(f'The mode of \"Network\" is \"close\", no action')

        return 

    def delete_instances(self, instances: List[Instance]) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                logger.info(f'Please set \"prototype\" first by using method: \"set_prototype(prototype)\".')
            if self.is_new:
                for index, instance in enumerate(instances):
                    if self.prototype == instance.prototype:
                        instance_identifier = instance.identifier
                        if instance_identifier in self._map:
                            self._map[instance_identifier].delete()
                        else:
                            logger.info(f'Skip, \"Instance\" not exists: {instance_identifier}.')
                    else:
                        logger.info(f'Skip No.{index}, \"Prototype\" of the \"Instance\" does not match the one of the \"Network\".')
            else:
                logger.info(f'The \"Network\" is in \"Old\" status, please re-New it.')
        else:
            logger.warn(f'The mode of \"Network\" is \"close\", no action')

        return

    def release(self, version: semantic_version.Version) -> None:
        assert self.mode_code == self.get_mode_code('open'), f'Can not release! The mode of \"Network\" is \"close\".'

        assert not self.is_fresh, f'Can not release! \"Network\" is not set.'

        for identifier, instance in self.map.items():
            instance.release(version)
            if identifier not in self.uniques and instance.is_new:
                self.uniques.append(identifier)
                self.instances.append(instance)

        self._map = dict()

        if version is not None:
            assert self.latest_version < version, (
                f'Version provided less than or equal to the latest version:\n'
                f'Provided: {version}'
                f'Latest: {self.latest_version}'
            )

            self._latest_version = version

            stamp = Stamp(
                str(version),
                self.checksum,
            )

            if stamp not in self._stamps:
                print(f'No Change on Network, no release.')
            else:
                self._stamps.add(stamp)

                if self.meta.release:
                    if self.is_old:
                        self.meta.set_retired(version)
                else:
                    if self.is_new:
                        self.meta.set_release(version)

        return