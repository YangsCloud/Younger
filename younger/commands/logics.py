#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-19 22:12:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-29 12:17:40
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click

from . import install_plugin_click_group


@click.group(name='logics')
def logics():
    pass


@install_plugin_click_group('ir', 'younger.logics', 'ir')
@click.group(name='ir')
def ir():
    pass


@install_plugin_click_group('core', 'younger.logics', 'core')
@click.group(name='core')
def core():
    pass


logics.add_command(ir)
logics.add_command(core)
