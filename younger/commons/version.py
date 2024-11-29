#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-28 09:35:59
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


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
