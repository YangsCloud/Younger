#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-10 14:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import enum

from onnx.defs import (
    ONNX_DOMAIN,
    ONNX_ML_DOMAIN,
    AI_ONNX_PREVIEW_TRAINING_DOMAIN,
)


class ONNX(enum.Enum):
    OP_DOMAIN = ONNX_DOMAIN or 'ai.onnx'
    OP_ML_DOMAIN = ONNX_ML_DOMAIN
    OP_PREVIEW_TRAINING_DOMAIN = AI_ONNX_PREVIEW_TRAINING_DOMAIN