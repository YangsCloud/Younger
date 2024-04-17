#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 23:32
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


class Schema(object):
    def __init__(self,
        **kwargs
    ) -> None:
        if kwargs.get('id', None) is not None:
            self.id = kwargs['id']

    def dict(self):
        items = dict()
        for key, value in self.__dict__.items():
            if key == 'id':
                continue
            if not key.startswith('_'):
                items[key] = value
        return items
