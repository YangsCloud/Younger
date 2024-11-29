#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-28 13:00:54
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click

from . import install_plugin_click_group


@click.group(name='apps')
def apps():
    pass


@install_plugin_click_group('dl', 'younger.apps', 'dl')
@click.group()
def dl():
    pass


apps.add_command(dl)