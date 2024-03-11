#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-11 21:26
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

class Model(object):
    def __init__(self,
        model_id: str | None = None,
        model_source: str | None = None,
        training_quality_metrics: dict = None,
        training_time_metrics: dict = None,
        inference_quality_metrics: dict = None, 
        inference_time_metrics: dict = None,
        model_likes: int | None = None,
        model_downloads: int | None = None,
        maintaining: bool | None= None,
        version: str | None = None,
        network_id: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if model_id is not None:
            self.model_id = model_id
        if model_source is not None:
            self.model_source = model_source

        if training_quality_metrics is not None:
            self.training_quality_metrics = training_quality_metrics
        if training_time_metrics is not None:
            self.training_time_metrics = training_time_metrics
        if inference_quality_metrics is not None:
            self.inference_quality_metrics = inference_quality_metrics
        if inference_time_metrics is not None:
            self.inference_time_metrics = inference_time_metrics

        if model_likes is not None:
            self.model_likes = model_likes
        if model_downloads is not None:
            self.model_downloads = model_downloads

        if maintaining is not None:
            self.maintaining = maintaining

        if version is not None:
            self.version = version
        
        if network_id is not None:
            self.network_id = network_id

    def dict(self):
        items = dict()
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                items[key] = value
        return items