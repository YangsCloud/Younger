#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-29 09:14:22
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click

from younger.commands.logics import logics
from younger.commands.tools import tools
from younger.commands.apps import apps


@click.group(name='younger')
def main():
    # naive_log(
    #     f'                                                                \n'
    #     f'                >   Welcome to use Younger!   <                 \n'
    #     f'----------------------------------------------------------------\n'
    #     f'Please use the following command to make the most of the system:\n'
    #     f'0. younger --help                                               \n'
    #     f'1. younger apps --help                                          \n'
    #     f'2. younger tools --help                                         \n'
    #     f'3. younger logics --help                                        \n'
    #     f'                                                                \n'
    # )
    # print(main.commands['logics'].commands['ir'].__dict__)
    pass


main.add_command(logics, name='logics')
main.add_command(tools, name='tools')
main.add_command(apps, name='apps')


if __name__ == '__main__':
    main()
