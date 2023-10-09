#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-14 09:07
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import semantic_version

from typing import Dict


class Stamp(object):
    def __init__(self, version: str, checksum: str) -> None:
        assert semantic_version.validate(version), f'The version provided must follow the SemVer 2.0.0 Specification.'
        self._version = semantic_version.Version(version)

        assert isinstance(checksum, str), f'Invalid type of checksum value, should be \"str\", instead \"{type(checksum)}\"'
        self._checksum = checksum

    @property
    def version(self) -> semantic_version.Version:
        return self._version

    @property
    def checksum(self) -> str:
        return self._checksum

    @property
    def dict(self) -> Dict:
        return dict(
            version = str(self.version),
            checksum = self.checksum,
        )

    def __repr__(self):
        return (
            f'Version: {str(self.version)}\n'
            f'Checksum: {self.checksum}\n'
        )

    def __hash__(self):
        hasher = hashlib.new('SHA256')
        hasher.update(self.checksum.encode('utf-8'))
        return hash(hasher.hexdigest())

    def __eq__(self, stamp: 'Stamp') -> bool:
        return self.checksum == stamp.checksum

    def __lt__(self, stamp: 'Stamp') -> bool:
        return self.version < stamp.version