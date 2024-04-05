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


class Constant(object):
    def __setattr__(self, attribute_name, attribute_value):
        assert attribute_name not in self.__dict__, f'Constant Name exists: \"{attribute_name}\"'

        self.__dict__[attribute_name] = attribute_value

    def __contains__(self, attribute_value):
        return attribute_value in self._values_

    @property
    def attributes(self) -> list:
        attributes = list()
        for key in self.__dict__.keys():
            if key != '_values_':
                attributes.append(key)
        return attributes

    def freeze(self):
        values = set()
        for value in self.__dict__.values():
            if isinstance(value, set) or isinstance(value, frozenset):
                values = values.union(value)
            else:
                values.add(value)

        self._values_ = values


class YOUNGER_HANDLE(Constant):
    def initialize(self) -> None:
        self.MainName = 'Younger'
        self.DatasetsName = 'Younger-Datasets'
        self.BenchmarksName = 'Younger-Benchmarks'
        self.ApplicationsName = 'Younger-Applications'


YoungerHandle = YOUNGER_HANDLE()
YoungerHandle.initialize()
YoungerHandle.freeze()