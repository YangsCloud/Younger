#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-03 14:09:57
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-28 13:00:20
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click

from . import install_plugin_click_group


@click.group(name='tools')
def tools():
    pass


@install_plugin_click_group('bench', 'younger.tools', 'bench')
@click.group()
def bench():
    pass


tools.add_command(bench)
