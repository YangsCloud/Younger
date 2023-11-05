#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-06 10:07
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import onnx
import pathlib
import semantic_version

from typing import Union

from youngbench.dataset.modules.meta import Meta

from youngbench.dataset.utils.io import read_json, write_json, load_onnx_model, save_onnx_model, check_onnx_model
from youngbench.dataset.utils.cache import get_cache_root


class Model(object):
    def __init__(self, onnx_model: Union[onnx.ModelProto, pathlib.Path] = None, version: semantic_version.Version = semantic_version.Version('0.0.0')) -> None:
        onnx_model = onnx_model or onnx.ModelProto()
        self._meta_filename = 'meta.json'
        self._meta = Meta()

        self._onnx_model_filename = 'model.onnx'
        self._onnx_model_cache_filepath = ''

        self._identifier = str()

        self._legacy = False

        self.set_onnx_model(onnx_model)
        self.release(version)

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def onnx_model(self) -> onnx.ModelProto:
        return load_onnx_model(self._onnx_model_cache_filepath)

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def status(self) -> int:
        status = ((self._onnx_model_cache_filepath == '') << 2) + (self.meta.release << 1) + (self.meta.retired or self._legacy)
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

    def copy(self) -> 'Model':
        model = Model()
        model._onnx_model_cache_filepath = self._onnx_model_cache_filepath
        model._identifier = self._identifier
        return model

    def load(self, model_dirpath: pathlib.Path) -> None:
        assert model_dirpath.is_dir(), f'There is no \"Model\" can be loaded from the specified directory \"{model_dirpath.absolute()}\".'
        meta_filepath = model_dirpath.joinpath(self._meta_filename)
        self._load_meta(meta_filepath)
        onnx_model_filepath = model_dirpath.joinpath(self._onnx_model_filename)
        self._load_onnx_model(onnx_model_filepath)
        return

    def save(self, model_dirpath: pathlib.Path) -> None:
        assert not model_dirpath.is_dir(), f'\"Model\" can not be saved into the specified directory \"{model_dirpath.absolute()}\".'
        onnx_model_filepath = model_dirpath.joinpath(self._onnx_model_filename)
        self._save_onnx_model(onnx_model_filepath)
        meta_filepath = model_dirpath.joinpath(self._meta_filename)
        self._save_meta(meta_filepath)
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

    def _load_onnx_model(self, onnx_model_filepath: pathlib.Path) -> None:
        assert onnx_model_filepath.is_file(), f'There is no \"ONNX Model\" can be loaded from the specified path \"{onnx_model_filepath.absolute()}\".'
        self._identifier = check_onnx_model(onnx_model_filepath)
        self._onnx_model_cache_filepath = onnx_model_filepath
        return

    def _save_onnx_model(self, onnx_model_filepath: pathlib.Path) -> None:
        assert not onnx_model_filepath.is_file(), f'\"ONNX Model\" can not be saved into the specified path \"{onnx_model_filepath.absolute()}\".'
        save_onnx_model(self.onnx_model, onnx_model_filepath)
        return

    def acquire(self, version: semantic_version.Version) -> 'Model':
        if (self.meta.release and self.meta.release_version <= version) and (not self.meta.retired or version < self.meta.retired_version):
            model = Model()
            model._identifier = self._identifier
            model._onnx_model_cache_filepath = self._onnx_model_cache_filepath
        else:
            model = None
        return model

    def set_onnx_model(self, onnx_model_handler: Union[onnx.ModelProto, pathlib.Path]) -> None:
        assert isinstance(onnx_model_handler, onnx.ModelProto) or isinstance(onnx_model_handler, pathlib.Path), f'\"Model\" must be an ONNX Model Proto (onnx.ModelProto) or a Path (pathlib.Path) instead \"{type(onnx_model_handler)}\"!'
        if self.is_fresh:
            identifier = check_onnx_model(onnx_model_handler)
            if isinstance(onnx_model_handler, onnx.ModelProto) and identifier:
                cache_root = get_cache_root()
                self._identifier = identifier
                self._onnx_model_cache_filepath = cache_root.joinpath(self._identifier)
                save_onnx_model(onnx_model_handler, self._onnx_model_cache_filepath)
            if isinstance(onnx_model_handler, pathlib.Path) and identifier:
                self._identifier = identifier
                self._onnx_model_cache_filepath = onnx_model_handler
        return

    def release(self, version: semantic_version.Version) -> None:
        if self.is_fresh or version == semantic_version.Version('0.0.0'):
            return

        if self.meta.release:
            if self.is_old:
                self.meta.set_retired(version)
        else:
            if self.is_new:
                self.meta.set_release(version)
        return