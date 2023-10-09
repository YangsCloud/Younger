#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-06 08:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import networkx


def vis(prototype):
    import matplotlib.pyplot as plt
    networkx.draw(prototype)
    import tempfile
    new_file = tempfile.NamedTemporaryFile(prefix='youngbench_', suffix='.jpg', delete=False)
    plt.savefig(new_file.name)
    print(f'See visualization of the \"Prototype\" at {new_file.name}')