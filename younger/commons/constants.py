#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-27 11:19:10
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


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
        self.AppName = 'Younger-App'
        self.ToolName = 'Younger-Tool'
        self.LogicName = 'Younger-Logic'

YoungerHandle = YOUNGER_HANDLE()
YoungerHandle.initialize()
YoungerHandle.freeze()