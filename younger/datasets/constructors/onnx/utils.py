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


from typing import Any
from onnx import hub, ModelProto


def get_onnx_model_infos() -> list[dict[str, Any]]:
    model_infos = dict()
    for model in sorted(hub.list_models(), key=lambda x: x.metadata['model_bytes']):
        model_id = model.model
        model_info = model_infos.get(model_id, list())
        model_info.append(
            dict(
                tags = list(model.tags),
                meta = model.metadata,
                path = model.model_path,
                hash = model.model_sha,
                opset = model.opset,
                raw = model.raw_model_info,
            )
        )
        model_infos[model_id] = model_info
    model_infos = [dict(id=model_id, variations=model_info) for model_id, model_info in model_infos.items()]
    return model_infos


def get_onnx_model_info(model_id: str) -> dict[str, Any]:
    model_info = list()
    for model in hub.list_models(model=model_id):
        model_info.append(
            dict(
                tags = list(model.tags),
                meta = model.metadata,
                path = model.model_path,
                hash = model.model_sha,
                opset = model.opset,
                raw = model.raw_model_info,
            )
        )
    model_info = dict(id=model_id, variations=model_info)
    return model_info


def get_onnx_model_ids() -> list[str]:
    model_infos = get_onnx_model_infos()
    model_ids = [model_info['id'] for model_info in model_infos]
    return model_ids


def get_opset_version(model: ModelProto):
    for opset_info in model.opset_import:
        if opset_info.domain == "":
            opset_version = opset_info.version
            break
    return opset_version
