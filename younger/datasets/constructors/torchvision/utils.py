#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-12 08:56
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchvision

from typing import Any


def get_torchvision_model_infos() -> list[dict[str, Any]]:
    model_infos = list()
    for model_id in torchvision.models.list_models():
        model_infos.append(
            dict(
                id = model_id,
                weights = {weights_enum.name: weights_enum.url for weights_enum in torchvision.models.get_model_weights(model_id)},
                metrics = {weights_enum.name: weights_enum.meta['_metrics'] for weights_enum in torchvision.models.get_model_weights(model_id)}
            )
        )
    return model_infos


def get_torchvision_model_info(model_id: str) -> dict[str, Any]:
    model_info = dict(
        id = model_id,
        weights = {weights_enum.name: weights_enum.url for weights_enum in torchvision.models.get_model_weights(model_id)},
        metrics = {weights_enum.name: weights_enum.meta['_metrics'] for weights_enum in torchvision.models.get_model_weights(model_id)}
    )
    return model_info


def get_torchvision_model_ids() -> list[str]:
    model_infos = get_torchvision_model_infos()
    model_ids = [model_info['id'] for model_info in model_infos]
    return model_ids


def get_torchvision_model_types() -> dict[str, str]:
    known_model_types = {
        'torchvision.models': 'classification',
        'torchvision.models.segmentation': 'segmentation',
        'torchvision.models.detection': 'detection',
        'torchvision.models.video': 'video',
        'torchvision.models.quantization': 'quantization',
        'torchvision.models.optical_flow': 'optical_flow',
    }
    model_types: dict[str, str] = dict()
    model_ids = get_torchvision_model_ids()
    model_modules: set[str] = set()
    for model_id in model_ids:
        model_module = get_torchvision_model_module(model_id)
        model_modules.add(model_module)
    
    for model_module in model_modules:
        if model_module not in known_model_types:
            model_type = '_unknown_'
        else:
            model_type = known_model_types[model_module]
        model_types[model_module] = model_type

    return model_types


def get_torchvision_model_module(model_id: str) -> str:
    model_builder = torchvision.models.get_model_builder(model_id)
    return model_builder.__module__.rpartition('.')[0]


def get_torchvision_model_type(model_id: str) -> str:
    model_module = get_torchvision_model_module(model_id)
    model_types = get_torchvision_model_types()
    return model_types[model_module]


def get_torchvision_model_input(model_id: str) -> torch.Tensor | tuple[torch.Tensor] | None:
    general_image_input = torch.randn(1, 3, 224, 224)
    optical_image_input = torch.randn(1, 3, 128, 128)
    model_inputs = dict(
        classification = general_image_input,
        segmentation = general_image_input,
        detection = general_image_input,
        quantization = general_image_input,
        optical_flow = (optical_image_input, optical_image_input),
        video = None,
        _unknown_ = None,
    )
    model_type = get_torchvision_model_type(model_id)
    return model_inputs[model_type]