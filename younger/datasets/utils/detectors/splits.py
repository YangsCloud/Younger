#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-14 11:38
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


splits = dict(
    train = set([
        'train', 'training',
    ]),
    valid = set([
        'dev', 'val', 'valid', 'validate', 'validation', 'eval', 'evaluate', 'evaluation',
    ]),
    test = set([
        'test', 'testing',
    ])
)


def detect_split(string: str) -> str:
    split = ''
    for word in string.split():
        if word in splits['test']:
            return 'test'
        elif word in splits['valid']:
            return 'valid'
        elif word in splits['train']:
            return 'train'
    return split