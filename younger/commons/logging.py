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


logging_level = dict(
    INFO = logging.INFO,
    WARN = logging.WARN,
    ERROR = logging.ERROR,
    DEBUG = logging.DEBUG,
    FATAL = logging.FATAL,
    NOTSET = logging.NOTSET
)


logger_dict = dict()


def get_logger(name: str) -> logging.Logger:
    try:
        logger = logger_dict[name]
    except Exception as error:
        logger = set_logger(name)
        logger.warn(f'Logger: \'{name}\' Does Not Exist. Now Using Default Logger. Error: {error}')
    
    return logger


def set_logger(
    name: str,
    mode: Literal['both', 'file', 'console'] = 'both',
    logging_level: int = logging.INFO,
    logging_filepath: pathlib.Path | None = None,
):
    assert mode in {'both', 'file', 'console'}, f'Not Support The Logging Mode - \'{mode}\'.'

    logging_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    logger.handlers.clear()

    if mode in {'both', 'file'}:
        if logging_filepath is None:
            logging_dirpath = pathlib.Path(os.getcwd())
            logging_filename = 'younger.log'
            logging_filepath = logging_dirpath.joinpath(logging_filename)
            print(f'Logging filepath is not specified, logging file will be saved in the working directory: \'{logging_dirpath}\', filename: \'{logging_filename}\'')
        else:
            logging_dirpath = logging_filepath.parent
            logging_filename = logging_filepath.name
            logging_filepath = str(logging_filepath)
            print(f'Logging file will be saved in the directory: \'{logging_dirpath}\', filename: \'{logging_filename}\'')

        file_handler = logging.FileHandler(logging_filepath, mode='a', encoding='utf-8')
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(logging_formatter)
        logger.addHandler(file_handler)

    if mode in {'both', 'console'}:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(logging_formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger_dict[name] = logger

    logger.info(f'Logger - Name: {name}; Mode: {mode}; Level: {logging_level}.')

    return logger