#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-15 16:54
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import semantic_version

from typing import Dict

from youngbench.dataset.utils.io import hash_string


class Meta(object):
    def __init__(
            self,
            private: bool = False,
            release: bool = False,
            retired: bool = False,
            release_version: str = None,
            retired_version: str = None,
            **metrics
    ) -> None:
        if release:
            assert semantic_version.validate(release_version), f'The release version provided must follow the SemVer 2.0.0 Specification.'
            release_version = semantic_version.Version(release_version)
        else:
            release_version = None

        if retired:
            assert release, f'Can not set \"retired\" while not \"release\".'
            assert semantic_version.validate(retired_version), f'The retired version provided must follow the SemVer 2.0.0 Specification.'
            retired_version = semantic_version.Version(retired_version)
        else:
            retired_version = None

        self._private = private
        self._release = release
        self._retired = retired
        self._release_version = release_version
        self._retired_version = retired_version
        self._metrics = metrics

    @property
    def private(self) -> bool:
        return self._private

    @property
    def release(self) -> bool:
        return self._release

    @property
    def retired(self) -> bool:
        return self._retired

    @property
    def release_version(self) -> semantic_version.Version:
        return self._release_version

    @property
    def retired_version(self) -> semantic_version.Version:
        return self._retired_version

    @property
    def metrics(self) -> Dict:
        return self._metrics

    @property
    def dict(self) -> Dict:
        return dict(
            private = self.private,
            release = self.release,
            retired = self.retired,
            release_version = str(self.release_version),
            retired_version = str(self.retired_version),
            metrics = self.metrics
        )

    @property
    def identifier(self) -> str:
        flags = self.dict
        flags.pop('metrics')
        return hash_string(str(tuple(sorted(flags.items()))))

    @property
    def on_the_fly(self) -> str:
        return ((not self.release) and (not self.retired))

    def set_release(self, release_version: semantic_version.Version) -> None:
        assert isinstance(release_version, semantic_version.Version), f'The release version provided must follow the SemVer 2.0.0 Specification.'
        assert not self.release, f'Object already has been released!'

        self._release = True
        self._release_version = release_version

    def set_retired(self, retired_version: semantic_version.Version) -> None:
        assert isinstance(retired_version, semantic_version.Version), f'The retired version provided must follow the SemVer 2.0.0 Specification.'
        assert not self.retired, f'Object already has been retired!'
        assert self.release, f'Can not retire a Object that has not been released!'

        assert self.release_version < retired_version, (
            f'Invalid retired version, it should be larger than the release version:\n'
            f'Release={self.release_version}\n'
            f'Retired={retired_version}'
        )
        self._retired = True
        self._retired_version = retired_version