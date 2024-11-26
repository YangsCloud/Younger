#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-06 16:53
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from younger.applications.models.napp_gat_base import NAPPGATBase
from younger.applications.models.napp_gin_base import NAPPGINBase
from younger.applications.models.napp_gat_vary_v1 import NAPPGATVaryV1
from younger.applications.models.glass import GLASS
from younger.applications.models.link_prediction import GCN_LP, GAT_LP, SAGE_LP, Encoder_LP
from younger.applications.models.node_prediction import GCN_NP, GIN_NP, GAT_NP, SAGE_NP, Encoder_NP, LinearCls
from younger.applications.models.embedding import MAEGIN