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
from youngbench.dataset.modules.network import Network

from youngbench.dataset.utils import hash_strings, read_json, write_json, create_dir
from youngbench.dataset.logging import logger


class Dataset(object):
    mode_codes = dict(
        close = 0B0,
        open = 0B1,
    )

    def __init__(self) -> None:
        self._stamps_filename = 'stamps.json'
        self._stamps = set()

        self._uniques_filename = 'uniques.json'
        self._uniques = list()

        self._networks_dirname = 'networks'
        self._networks = list()

        self._latest_version = semantic_version.Version('0.0.0')

        self._mode_code = 0B0
        self._map = dict()

    @property
    def stamps(self) -> Set[Stamp]:
        return self._stamps

    @property
    def uniques(self) -> List[str]:
        return self._uniques

    @property
    def networks(self) -> List[Network]:
        return self._networks

    @property
    def latest_version(self) -> semantic_version.Version:
        return self._latest_version

    @property
    def checksum(self) -> str:
        if len(self.map) != 0:
            logger.warn(f'There are still unreleased \"Network\"s; the checksum may be incorrect.')
        ids = list()
        for network in self.networks:
            ids.append(network.identifier)
            ids.append(network.meta.identifier)

        return hash_strings(ids)

    @property
    def mode_code(self) -> bool:
        return self._mode_code

    @property
    def map(self) -> Dict[str, Network]:
        return self._map

    def get_mode_code(self, mode_pattern: str) -> int:
        return self.__class__.mode_codes[mode_pattern]

    def set_mode_code(self, mode_pattern: str) -> None:
        self._mode_code = self.__class__.mode_codes[mode_pattern]

        return

    def open(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'The mode of \"Dataset\" is \"open\", do not need open it again.')
        self.set_mode_code('open')
        self._map = dict()
        for identifier, network in zip(self.uniques, self.networks):
            network.open()
            self._map[identifier] = network
        return 

    def __enter__(self) -> None:
        self.open()
        return self

    def close(self) -> None:
        if self.mode_code == self.get_mode_code('close'):
            logger.warn(f'The mode of \"Dataset\" is \"close\", do not need close it again.')
        self.set_mode_code('close')
        if len(self._map) != 0:
            logger.warn(f'There are still unreleased \"Network\"s; all are lost.')
        for identifier, network in zip(self.uniques, self.networks):
            network.close()
        self._map = dict()

        return

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def load(self, dataset_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Dataset\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert dataset_dirpath.is_dir(), f'There is no \"Dataset\" can be loaded from the specified directory \"{dataset_dirpath.absolute()}\".'
            logger.info(f'Now loading \"Dataset\" from {dataset_dirpath.absolute()} ...')
            stamps_filepath = dataset_dirpath.joinpath(self._stamps_filename)
            self._load_stamps(stamps_filepath)
            uniques_filepath = dataset_dirpath.joinpath(self._uniques_filename)
            self._load_uniques(uniques_filepath)
            networks_dirpath = dataset_dirpath.joinpath(self._networks_dirname)
            self._load_networks(networks_dirpath)

        return

    def save(self, dataset_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Dataset\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not dataset_dirpath.is_dir(), f'\"Dataset\" can not be saved into the specified directory \"{dataset_dirpath.absolute()}\".'
            create_dir(dataset_dirpath)
            logger.info(f'Now saving \"Dataset\" into {dataset_dirpath.absolute()} ...')
            networks_dirpath = dataset_dirpath.joinpath(self._networks_dirname)
            self._save_networks(networks_dirpath)
            uniques_filepath = dataset_dirpath.joinpath(self._uniques_filename)
            self._save_uniques(uniques_filepath)
            stamps_filepath = dataset_dirpath.joinpath(self._stamps_filename)
            self._save_stamps(stamps_filepath)

        return

    def _load_stamps(self, stamps_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Dataset\" is \"open\".')

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
            logger.warn(f'Can not save! The mode of \"Dataset\" is \"open\".')

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
            logger.warn(f'Can not load! The mode of \"Dataset\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert uniques_filepath.is_file(), f'There is no \"Unique\"s can be loaded from the specified path \"{uniques_filepath.absolute()}\".'

            self._uniques = read_json(uniques_filepath)
            assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'

        return

    def _save_uniques(self, uniques_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Dataset\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not uniques_filepath.is_file(), f'\"Unique\"s can not be saved into the specified path \"{uniques_filepath.absolute()}\".'

            uniques_filepath.touch()
            assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
            write_json(self._uniques, uniques_filepath)

        return

    def _load_networks(self, networks_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not load! The mode of \"Dataset\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert networks_dirpath.is_dir(), f'There is no \"Network\"s can be loaded from the specified directory \"{networks_dirpath.absolute()}\".'

            for index, identifier in enumerate(self._uniques):
                network_dirpath = networks_dirpath.joinpath(f'{index}-{identifier}')
                network = Network()
                network.load(network_dirpath)
                self._networks.append(network)

        return

    def _save_networks(self, networks_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Dataset\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not networks_dirpath.is_dir(), f'\"Network\"s can not be saved into the specified directory \"{networks_dirpath.absolute()}\".'

            create_dir(networks_dirpath)
            for index, identifier in enumerate(self._uniques):
                network_dirpath = networks_dirpath.joinpath(f'{index}-{identifier}')
                network = self._networks[index]
                network.save(network_dirpath)

        return

    def acquire(self, version: semantic_version.Version) -> 'Dataset':
        assert self.mode_code == self.get_mode_code('close'), f'Can not release! The mode of \"Dataset\" is \"open\".'

        dataset = Dataset()
        dataset.open()
        for network in self.networks:
            dataset.add(network.acquire(version))

        dataset.release(version)
        dataset.close()
        return dataset

    def check(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'No Check! The mode of \"Dataset\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            logger.info(f'Begin Check!')
            latest_version = semantic_version.Version('0.0.0')
            for stamp in self.stamps:
                latest_version = max(latest_version, stamp.version)
            assert self.latest_version == latest_version, f'\"Latest version\" incorrect, it should be \"{latest_version}\" instead \"{self.latest_version}\"'

            assert len(self.uniques) == len(self.networks), f'The number of \"Network\"s does not match the number of \"Unique\"s.'

            for identifier, network in zip(self.uniques, self.networks):
                assert identifier == network.identifier, f'The \"Identifier={network.identifier}\" of \"Network\" does not match \"Unique={identifier}\" '
                network.check()

            for stamp in self.stamps:
                dataset = self.acquire(stamp.version)
                assert stamp.checksum == dataset.checksum, f'The \"Checksum={dataset.checksum}\" of \"Dataset\" (Version={stamp.version}) does not match \"Stamp={stamp.checksum}\"'

        return

    def add(self, network: Network) -> None:
        if self.mode_code == self.get_mode_code('open'):
            network_identifier = network.identifier
            if network_identifier not in self._map:
                self._map[network_identifier] = Network()
                self._map[network_identifier].open()
                self._map[network_identifier].set_prototype(network.prototype)
            self._map[network_identifier].add()
            self._map[network_identifier].add_instances(network.instances)
        else:
            logger.warn(f'The mode of \"Dataset\" is \"close\", no action')

        return

    def delete(self, network: Network) -> None:
        if self.mode_code == self.get_mode_code('open'):
            network_identifier = network.identifier
            if network_identifier in self._map:
                if len(network.instances) == 0:
                    self._map[network_identifier].delete()
                else:
                    self._map[network_identifier].delete_instances(network.instances)
            else:
                logger.info(f'No such \"Network\", skip \"delete\" action.')
        else:
            logger.warn(f'The mode of \"Dataset\" is \"close\", no action')

        return

    def release(self, version: semantic_version.Version) -> None:
        assert self.mode_code == self.get_mode_code('open'), f'Can not release! The mode of \"Dataset\" is \"close\".'

        for identifier, network in self.map.items():
            network.release(version)
            if identifier not in self.uniques and network.is_new:
                self.uniques.append(identifier)
                self.networks.append(network)

        self._map = dict()

        assert self.latest_version < version, (
            f'Version provided less than or equal to the latest version:\n'
            f'Provided: {version}'
            f'Latest: {self.latest_version}'
        )

        stamp = Stamp(
            str(version),
            self.checksum,
        )

        if stamp not in self._stamps:
            print(f'No Change on Dataset, no release.')
        else:
            self._stamps.add(stamp)

            self._latest_version = version

        return