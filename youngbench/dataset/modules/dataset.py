#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-11 08:16
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import semantic_version

from typing import Set, List, Dict

from youngbench.dataset.modules.stamp import Stamp
from youngbench.dataset.modules.instance import Instance

from youngbench.dataset.utils.io import hash_strings, read_json, write_json
from youngbench.logging import logger


class Dataset(object):
    def __init__(self, instances: List[Instance] = None, version: semantic_version.Version = semantic_version.Version('0.0.0')) -> None:
        instances = instances or list()

        self._stamps_filename = 'stamps.json'
        self._stamps = set()

        self._uniques_filename = 'uniques.json'
        self._uniques = list()

        self._instances_dirname = 'instances'
        self._instances = dict()

        self._mode = None

        self.insert_instances(instances)
        self.release(version)

    @property
    def stamps(self) -> Set[Stamp]:
        return self._stamps

    @property
    def uniques(self) -> List[str]:
        return self._uniques

    @property
    def instances(self) -> Dict[str, Instance]:
        return self._instances

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
        for identifier in self.uniques:
            instance = self.instances[identifier]
            if instance.is_release:
                ids.append(instance.checksum)
        return hash_strings(ids)

    def load(self, dataset_dirpath: pathlib.Path) -> None:
        assert dataset_dirpath.is_dir(), f'There is no \"Dataset\" can be loaded from the specified directory \"{dataset_dirpath.absolute()}\".'
        logger.info(f' = [YBD] = Loading Dataset @ {dataset_dirpath}...')
        stamps_filepath = dataset_dirpath.joinpath(self._stamps_filename)
        self._load_stamps(stamps_filepath)
        uniques_filepath = dataset_dirpath.joinpath(self._uniques_filename)
        self._load_uniques(uniques_filepath)
        instances_dirpath = dataset_dirpath.joinpath(self._instances_dirname)
        self._load_instances(instances_dirpath)
        return

    def save(self, dataset_dirpath: pathlib.Path) -> None:
        assert not dataset_dirpath.is_dir(), f'\"Dataset\" can not be saved into the specified directory \"{dataset_dirpath.absolute()}\".'
        logger.info(f' = [YBD] = Saving Dataset @ {dataset_dirpath}...')
        instances_dirpath = dataset_dirpath.joinpath(self._instances_dirname)
        self._save_instances(instances_dirpath)
        uniques_filepath = dataset_dirpath.joinpath(self._uniques_filename)
        self._save_uniques(uniques_filepath)
        stamps_filepath = dataset_dirpath.joinpath(self._stamps_filename)
        self._save_stamps(stamps_filepath)
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

    def _load_instances(self, instances_dirpath: pathlib.Path) -> None:
        assert instances_dirpath.is_dir(), f'There is no \"Instance\"s can be loaded from the specified directory \"{instances_dirpath.absolute()}\".'
        logger.info(f' = [YBD] = Loading Instances ...')
        for index, identifier in enumerate(self._uniques):
            logger.info(f' = [YBD] = No.{index} Instance: {identifier}')
            instance_dirpath = instances_dirpath.joinpath(f'{index}-{identifier}')
            self._instances[identifier] = Instance()
            self._instances[identifier].load(instance_dirpath)
        return

    def _save_instances(self, instances_dirpath: pathlib.Path) -> None:
        assert not instances_dirpath.is_dir(), f'\"Instance\"s can not be saved into the specified directory \"{instances_dirpath.absolute()}\".'
        logger.info(f' = [YBD] = Saving Instances ...')
        for index, identifier in enumerate(self._uniques):
            logger.info(f' = [YBD] = No.{index} Instance: {identifier}')
            instance_dirpath = instances_dirpath.joinpath(f'{index}-{identifier}')
            instance = self._instances[identifier]
            instance.save(instance_dirpath)
        return

    def acquire(self, version: semantic_version.Version, silence: bool = False) -> 'Dataset':
        if not silence:
            logger.info(f' = [YBD] = Acquiring Dataset: version = {version}...')
        dataset = Dataset()
        for index, identifier in enumerate(self._uniques):
            instance = self._instances[identifier].acquire(version, silence=silence)
            if instance is not None:
                if not silence:
                    logger.info(f' = [YBD] = Acquired No.{index} Instance: {identifier}')
                dataset._instances[identifier] = instance
        dataset.release(version=version)
        return dataset

    def check(self) -> None:
        # Check Instances
        assert len(self.uniques) == len(self.instances), f'The number of \"Instance\"s does not match the number of \"Unique\"s.'
        for identifier in self.uniques:
            instance = self.instances[identifier]
            assert identifier == instance.identifier, f'The \"Identifier={instance.identifier}\" of \"Instance\" does not match \"Unique={identifier}\" '
            instance.check()
        # Check Stamps
        for stamp in self.stamps:
            dataset = self.acquire(stamp.version, silence=True)
            assert stamp.checksum == dataset.checksum, f'The \"Checksum={dataset.checksum}\" of \"Dataset\" (Version={stamp.version}) does not match \"Stamp={stamp.checksum}\"'
        return

    def size(self) -> Dict[str, int]:
        instance_number = len(self.instances)
        model_number = 0
        for instance in self.instances.values():
            model_number += len(instance)
        return dict(
            instance_number = instance_number,
            model_number = model_number,
        )

    def insert(self, instance: Instance) -> bool:
        new_instance = self._instances.get(instance.identifier, Instance())
        flag = new_instance.setup_prototype(instance.prototype)
        flags_sum = new_instance.insert_networks(instance.networks.values())
        self._instances[new_instance.identifier] = new_instance
        return (flags_sum > 0) or flag

    def delete(self, instance: Instance) -> bool:
        old_instance = self._instances.get(instance.identifier, Instance())
        flags_sum = old_instance.delete_networks(instance.networks.values())
        flag = old_instance.clear_prototype(instance.prototype)
        return (flags_sum > 0) or flag

    def insert_instances(self, instances: List[Instance]) -> int:
        flags = list()
        for instance in instances:
            flags.append(self.insert(instance))
        return sum(flags)

    def delete_instances(self, instances: List[Instance]) -> int:
        flags = list()
        for instance in instances:
            flags.append(self.delete(instance))
        return sum(flags)

    def release(self, version: semantic_version.Version) -> None:
        if version == semantic_version.Version('0.0.0'):
            return
        assert self.latest_version < version, (
            f'Version provided less than or equal to the latest version:\n'
            f'Provided: {version}\n'
            f'Latest: {self.latest_version}'
        )

        for identifier, instance in self._instances.items():
            if instance.is_external:
                if instance.is_new:
                    self._uniques.append(identifier)
                if instance.is_old:
                    self._instances.pop(identifier)
            instance.release(version)

        stamp = Stamp(
            str(version),
            self.checksum,
        )
        if stamp in self._stamps:
            return
        else:
            self._stamps.add(stamp)
        return