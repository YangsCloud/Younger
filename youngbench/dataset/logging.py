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


from yoolkit.logging import setup_logger, logging_level

logger = setup_logger('YBDataset', logging_level=logging_level['INFO'])
#logger = setup_logger('YBDataset', logging_level=logging_level['WARN'])
#logger = setup_logger('YBDataset', logging_level=logging_level['ERROR'])
#logger = setup_logger('YBDataset', logging_level=logging_level['DEBUG'])