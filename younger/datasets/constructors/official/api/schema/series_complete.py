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


class SeriesCompleteItem(Schema):
    def __init__(self,
        instance_name: str | None = None,
        model_name: str | None = None,
        model_source: str | None = None,
        model_part: str | None = None,
        node_number: int | None = None,
        edge_number: int | None = None,
        since_version: str | None = None,
        paper: str | None = None,
        status: str | None = None,
        instance_tgz: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if instance_name is not None:
            self.instance_name = instance_name
        if model_name is not None:
            self.model_name = model_name
        if model_source is not None:
            self.model_source = model_source
        if model_part is not None:
            self.model_part = model_part

        if node_number is not None:
            self.node_number = node_number
        if edge_number is not None:
            self.edge_number = edge_number

        if since_version is not None:
            self.since_version = since_version
        if paper is not None:
            self.paper = paper
        if status is not None:
            self.status = status

        if instance_tgz is not None:
            self.instance_tgz = instance_tgz