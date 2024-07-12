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


class SeriesFilterItem(Schema):
    def __init__(self,
        instance_hash: str | None = None,
        node_number: int | None = None,
        edge_number: int | None = None,
        with_attributes: bool | None = None,
        since_version: str | None = None,
        paper: bool | None = None,
        status: str | None = None,
        instance_meta: str | None = None,
        instance_tgz: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if instance_hash is not None:
            self.instance_hash = instance_hash

        if node_number is not None:
            self.node_number = node_number
        if edge_number is not None:
            self.edge_number = edge_number

        if with_attributes is not None:
            self.with_attributes = with_attributes
        if since_version is not None:
            self.since_version = since_version
        if paper is not None:
            self.paper = paper
        if status is not None:
            self.status = status

        if instance_meta is not None:
            self.instance_meta = instance_meta
        if instance_tgz is not None:
            self.instance_tgz = instance_tgz