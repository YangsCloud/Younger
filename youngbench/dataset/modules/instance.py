#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-14 12:22
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import onnx
import pathlib
import networkx
import semantic_version

from youngbench.dataset.modules.meta import Meta
from youngbench.dataset.modules.prototype import Prototype

from youngbench.dataset.utils import read_json, write_json, hash_bytes, load_onnx_model, save_onnx_model, create_cache, remove_cache
from youngbench.dataset.logging import logger


class Instance(object):
    mode_codes = dict(
        close = 0B0,
        open = 0B1,
    )

    def __init__(self, model: onnx.ModelProto = None, version: semantic_version.Version = None) -> None:
        self._meta_filename = 'meta.json'
        self._meta = Meta()

        self._model_filename = 'model.onnx'
        self._model_filepath = ''
        self._model_cache = ''

        self._mode_code = 0B0
        self._legacy = False
        self._prototype = Prototype()

        if model:
            self.open()
            self.set_model(model)
            self.release(version)
            self.close()

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def model_filepath(self) -> str:
        return self._model_filepath

    @property
    def model_cache(self) -> str:
        return self._model_cache

    @property
    def model(self) -> onnx.ModelProto:
        if self._model_cache:
            return load_onnx_model(self.model_cache)
        else:
            return None

    @property
    def prototype(self) -> networkx.DiGraph:
        self._prototype.from_onnx_model(self.model)
        return self._prototype

    @property
    def identifier(self) -> str:
        return hash_bytes(self.model.SerializeToString())

    @property
    def mode_code(self) -> bool:
        return self._mode_code

    def get_mode_code(self, mode_pattern: str) -> int:
        return self.__class__.mode_codes[mode_pattern]

    def set_mode_code(self, mode_pattern: str) -> None:
        self._mode_code = self.__class__.mode_codes[mode_pattern]

        return

    def open(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'The mode of \"Instance\" is \"open\", do not need open it again.')
        self.set_mode_code('open')
        return 

    def __enter__(self) -> None:
        self.open()
        return self

    def close(self) -> None:
        if self.mode_code == self.get_mode_code('close'):
            logger.warn(f'The mode of \"Instance\" is \"close\", do not need close it again.')
        self.set_mode_code('close')

        return

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def status(self) -> int:
        status = ((self.model is None) << 2) + (self.meta.release << 1) + (self.meta.retired or self._legacy)
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

    def load(self, instance_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not lod! The mode of \"Instance\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert instance_dirpath.is_dir(), f'There is no \"Instance\" can be loaded from the specified directory \"{instance_dirpath.absolute()}\".'
            meta_filepath = instance_dirpath.joinpath(self._meta_filename)
            self._load_meta(meta_filepath)
            model_filepath = instance_dirpath.joinpath(self._model_filename)
            self._load_model(model_filepath)

        return

    def save(self, instance_dirpath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Instance\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not instance_dirpath.is_dir(), f'\"Instance\" can not be saved into the specified directory \"{instance_dirpath.absolute()}\".'
            instance_dirpath.mkdir()
            model_filepath = instance_dirpath.joinpath(self._model_filename)
            self._save_model(model_filepath)
            meta_filepath = instance_dirpath.joinpath(self._meta_filename)
            self._save_meta(meta_filepath)

        return


    def _load_meta(self, meta_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not lod! The mode of \"Instance\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert meta_filepath.is_file(), f'There is no \"Meta\" can be loaded from the specified path \"{meta_filepath.absolute()}\".'
            meta = read_json(meta_filepath)
            self._meta = Meta(**meta)

        return

    def _save_meta(self, meta_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Instance\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not meta_filepath.is_file(), f'\"Stamp\" can not be saved into the specified path \"{meta_filepath.absolute()}\".'
            write_json(self.meta.dict, meta_filepath)

        return

    def _load_model(self, model_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not lod! The mode of \"Instance\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert model_filepath.is_file(), f'There is no \"Model\" can be loaded from the specified path \"{model_filepath.absolute()}\".'
            model = load_onnx_model(model_filepath)
            self._model_filepath = model_filepath
            self._model_cache = model_filepath

        return

    def _save_model(self, model_filepath: pathlib.Path) -> None:
        if self.mode_code == self.get_mode_code('open'):
            logger.warn(f'Can not save! The mode of \"Instance\" is \"open\".')

        if self.mode_code == self.get_mode_code('close'):
            assert not model_filepath.is_file(), f'\"Model\" can not be saved into the specified path \"{model_filepath.absolute()}\".'
            save_onnx_model(self.model, model_filepath)

        return

    def set_model(self, model: onnx.ModelProto) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                assert isinstance(model, onnx.ModelProto), f'\"Model\" must be an ONNX Model Proto (onnx.ModelProto) instead \"{type(model)}\"!'
                self._model_cache = create_cache(model)
            else:
                logger.info(f'\"Model\" has already been set, \"set\" operation will not take effect.')
        else:
            logger.warn(f'The mode of \"Instance\" is \"close\", no action')

        return

    def clean_cache(self) -> None:
        if len(self._model_filepath) == 0:
            remove_cache(self.model, self.model_cache)
        return

    def add(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                logger.warn(f'Please set \"model\" first by using method: \"set_model(model)\".')
            if self.is_old:
                self.set_new()
        else:
            logger.warn(f'The mode of \"Instance\" is \"close\", no action')

        return

    def delete(self) -> None:
        if self.mode_code == self.get_mode_code('open'):
            if self.is_fresh:
                logger.warn(f'Please set \"model\" first by using method: \"set_model(model)\".')
            if self.is_new:
                self.set_old()
        else:
            logger.warn(f'The mode of \"Instance\" is \"close\", no action')

        return

    def release(self, version: semantic_version.Version) -> None:
        assert self.mode_code == self.get_mode_code('open'), f'Can not release! The mode of \"Instance\" is \"close\".'

        assert not self.is_fresh, f'Can not release! \"Instance\" is not set.'

        if version is not None:
            if self.meta.release:
                if self.is_old:
                    self.meta.set_retired(version)
            else:
                if self.is_new:
                    self.meta.set_release(version)

        return