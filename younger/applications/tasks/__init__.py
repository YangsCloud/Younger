#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-17 03:29
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from younger.applications.tasks.base_task import YoungerTask
from younger.applications.tasks.performance_prediction import PerformancePrediction
from younger.applications.tasks.block_embedding import BlockEmbedding
from younger.applications.tasks.node_prediction import NodePrediction
from younger.applications.tasks.link_prediction import LinkPridiction
from younger.applications.tasks.node_embedding import NodeEmbedding
from younger.applications.tasks.ssl_prediction import SSLPrediction


task_builders: dict[str, YoungerTask] = dict(
    performance_prediction = PerformancePrediction,
    block_embedding = BlockEmbedding,
    node_prediciton = NodePrediction,
    link_prediction = LinkPridiction,
    node_embedding = NodeEmbedding,
    ssl_prediction= SSLPrediction,
)