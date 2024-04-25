#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 23:22
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from younger.datasets.constructors.official.api.schema import Schema


class Model(Schema):
    def __init__(self,
        maintain: bool | None= None,
        model_id: str | None = None,
        model_source: str | None = None,
        raw_metrics: dict | None = None,
        training_metrics: dict | None = None,
        inference_metrics: dict | None = None, 
        version: str | None = None,
        model_likes: int | None = None,
        model_downloads: int | None = None,
        network_id: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if maintain is not None:
            self.maintain = maintain
        if model_id is not None:
            self.model_id = model_id
        if model_source is not None:
            self.model_source = model_source

        if raw_metrics is not None:
            self.raw_metrics = raw_metrics
        if training_metrics is not None:
            self.training_metrics = training_metrics
        if inference_metrics is not None:
            self.inference_metrics = inference_metrics

        if version is not None:
            self.version = version
        if model_likes is not None:
            self.model_likes = model_likes
        if model_downloads is not None:
            self.model_downloads = model_downloads

        if network_id is not None:
            self.network_id = network_id