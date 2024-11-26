#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-10 14:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

class Constant:
    __slots__ = ("_frozen_", "_attributes_")

    def __init__(self, **kwargs):
        object.__setattr__(self, "_attributes_",  dict())
        for key, value in kwargs.items():
            self._attributes_[key] = value
        object.__setattr__(self, "_frozen_",  False)

    def freeze(self):
        object.__setattr__(self, "_frozen_",  True)

    def __setattr__(self, key, value):
        if self._frozen_:
            raise AttributeError("Cannot modify a frozen Constant instance.")
        if key in self._attributes_:
            raise AttributeError(f"Constant Name exists: \"{key}\"")
        self._attributes_[key] = value

    def __getattr__(self, key):
        if key in self._attributes_:
            return self._attributes_[key]
        raise AttributeError(f"No such constant: \"{key}\"")

    @property
    def attributes(self):
        return list(self._attributes_.keys())


class YOUNGER_HANDLE(Constant):
    def initialize(self) -> None:
        self.MainName = 'Younger'
        self.DatasetsName = 'Younger-Datasets'
        self.BenchmarksName = 'Younger-Benchmarks'
        self.ApplicationsName = 'Younger-Applications'


YoungerHandle = YOUNGER_HANDLE()
YoungerHandle.initialize()
YoungerHandle.freeze()