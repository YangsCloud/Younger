#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-06 06:51
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import pathlib
import logging

from typing import Literal

from younger.commons.constants import YoungerHandle


logging_level = dict(
    INFO = logging.INFO,
    WARN = logging.WARN,
    ERROR = logging.ERROR,
    DEBUG = logging.DEBUG,
    FATAL = logging.FATAL,
    NOTSET = logging.NOTSET
)


logger_dict = dict()


def naive_log(message: str, silence: bool = False):
    if silence:
        return
    else:
        print(message)
        sys.stdout.flush()
        return


def get_logger(name: str) -> logging.Logger:
    try:
        logger = logger_dict[name]
    except Exception as error:
        naive_log(f'Logger: \'{name}\' Does Not Exist. Now Using Default Logger [Only Show On Console]. Warn: {error}')
        logger = set_logger(name, mode='console')
    
    return logger


def set_logger(
    name: str,
    mode: Literal['both', 'file', 'console'] = 'both',
    level: Literal['INFO', 'WARN', 'ERROR', 'DEBUG', 'FATAL', 'NOTSET'] = 'INFO',
    logging_filepath: pathlib.Path | None = None,
    show_setting_log: bool = True
):
    assert mode in {'both', 'file', 'console'}, f'Not Support The Logging Mode - \'{mode}\'.'
    assert level in {'INFO', 'WARN', 'ERROR', 'DEBUG', 'FATAL', 'NOTSET'}, f'Not Support The Logging Level - \'{level}\'.'

    logging_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging_level[level])

    logger.handlers.clear()

    if mode in {'both', 'file'}:
        if logging_filepath is None:
            logging_dirpath = pathlib.Path(os.getcwd())
            logging_filename = 'younger.log'
            logging_filepath = logging_dirpath.joinpath(logging_filename)
            naive_log(f'Logging filepath is not specified, logging file will be saved in the working directory: \'{logging_dirpath}\', filename: \'{logging_filename}\'', silence=not show_setting_log)
        else:
            logging_dirpath = logging_filepath.parent
            logging_filename = logging_filepath.name
            logging_filepath = str(logging_filepath)
            naive_log(f'Logging file will be saved in the directory: \'{logging_dirpath}\', filename: \'{logging_filename}\'', silence=not show_setting_log)

        file_handler = logging.FileHandler(logging_filepath, mode='a', encoding='utf-8')
        file_handler.setLevel(logging_level[level])
        file_handler.setFormatter(logging_formatter)
        logger.addHandler(file_handler)

    if mode in {'both', 'console'}:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging_level[level])
        console_handler.setFormatter(logging_formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger_dict[name] = logger

    naive_log(f'Logger: \'{name}\' - \'{mode}\' - \'{level}\'', silence=not show_setting_log)

    return logger


def use_logger(name: str):
    global logger
    logger = get_logger(name)


set_logger(YoungerHandle.MainName, mode='console', level='INFO', show_setting_log=False)

use_logger(YoungerHandle.MainName)

logger = get_logger(YoungerHandle.MainName)