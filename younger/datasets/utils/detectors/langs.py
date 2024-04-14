#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-13 18:47
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from isocodes import languages, extendend_languages


program_langs = set([
    'bash',
    'cplusplus',
    'csharp',
    'd',
    'go',
    'java',
    'javascript',
    'julia',
    'lua',
    'perl',
    'php',
    'prompted',
    'python',
    'r',
    'racket',
    'ruby',
    'rust',
    'scala',
    'swift',
    'typescript',
])


def detect_program_langs(string: str):
    plangs = list()
    for word in string.split():
        if word in program_langs:
            if word in {'c++', 'cpp'}:
                word = 'cplusplus'
            if word in {'c#'}:
                word = 'csharp'
            if word in program_langs:
                plangs.append(word)
    
    return plangs


def detect_natural_langs(string: str):
    nlangs = list()
    for word in string.split():
        if len(word) == 2:
            kwargs = {'alpha_2': word}
        elif len(word) == 3:
            kwargs = {'alpha_3': word}
        else:
            kwargs = {'name': word.title()}
        item = languages.get(**kwargs) or extendend_languages.get(**kwargs)
        if item:
            nlangs.append(item['alpha_3'])
    return nlangs