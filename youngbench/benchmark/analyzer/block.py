#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-10 11:47
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import networkx

from typing import List, Dict, Tuple

from youngbench.dataset.modules import Dataset, Prototype


def get_blocks_of_prototype(prototype: Prototype, before: int, after: int) -> List[Dict[str, Tuple[Prototype, ]]]:
    pass
    x = networkx.DiGraph()
    i  = x.nodes[1]
    i.d