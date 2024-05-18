#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 20:43
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import semantic_release


def check_semantic(version: str) -> bool:
    try:
        semantic_release.Version.parse(version)
        result = True
    except Exception as exception:
        result = False
    return result


def str_to_sem(str_ver: str) -> semantic_release.Version:
    sem_ver = semantic_release.Version.parse(version_str=str_ver)
    return sem_ver


def sem_to_str(sem_ver: semantic_release.Version) -> str:
    str_ver = str(sem_ver)
    return str_ver