#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-05 01:24
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from onnx import hub, ModelProto


def get_onnx_model_infos():
    onnx_model_infos = sorted(hub.list_models(), key=lambda x: x.metadata['model_bytes'])
    return onnx_model_infos


def get_opset_version(model: ModelProto):
    for opset_info in model.opset_import:
        if opset_info.domain == "":
            opset_version = opset_info.version
            break
    return opset_version