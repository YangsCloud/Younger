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


from typing import Callable

from younger.commons.io import load_json, save_json
from younger.commons.version import semantic_release, check_semantic, str_to_sem, sem_to_str


class Meta(object):
    def __init__(
            self,
            fresh_checker: Callable[[], bool],
            release: bool = False,
            release_version: str | None = None,
            retired: bool = False,
            retired_version: str | None = None,
            private: bool = False,
    ) -> None:
        if release:
            assert check_semantic(release_version), f'The release version provided must follow the SemVer 2.0.0 Specification.'
            release_version = str_to_sem(release_version)
        else:
            release_version = None

        if retired:
            assert release, f'Can not set \"retired\" while not \"release\".'
            assert check_semantic(retired_version), f'The retired version provided must follow the SemVer 2.0.0 Specification.'
            retired_version = str_to_sem(retired_version)
        else:
            retired_version = None

        self._fresh_checker = fresh_checker

        self._release = release
        self._release_version = release_version
        self._retired = retired
        self._retired_version = retired_version

        self._private = private

        self._legacy = False
        return

    @property
    def release(self) -> bool:
        return self._release

    @property
    def release_version(self) -> semantic_release.Version | None:
        return self._release_version

    @property
    def retired(self) -> bool:
        return self._retired

    @property
    def retired_version(self) -> semantic_release.Version | None:
        return self._retired_version

    @property
    def private(self) -> bool:
        return self._private

    def load(self, meta_filepath) -> None:
        info: dict = load_json(meta_filepath)

        if info.get('release', False):
            self._release = True
            self._release_version = str_to_sem(info['release_version'])
        if info.get('retired', False):
            self._retired = True
            self._retired_version = str_to_sem(info['retired_version'])
        if info.get('private', False):
            self._private = True

        return

    def save(self, meta_filepath) -> None:
        info: dict = dict()

        if self._release:
            info.update(release=True)
            info.update(release_version=sem_to_str(self._release_version))
        if self._retired:
            info.update(retired=True)
            info.update(retired_version=sem_to_str(self._retired_version))
        if self._private:
            info.update(private=True)

        save_json(info, meta_filepath)
        return

    def set_release(self, release_version: semantic_release.Version) -> None:
        assert not self.is_fresh, f'Object is fresh.'
        assert isinstance(release_version, semantic_release.Version), f'The release version provided must follow the SemVer 2.0.0 Specification.'
        assert not self.release, f'Object already has been released!'

        self._release = True
        self._release_version = release_version
        return

    def set_retired(self, retired_version: semantic_release.Version) -> None:
        assert not self.is_fresh, f'Object is fresh.'
        assert isinstance(retired_version, semantic_release.Version), f'The retired version provided must follow the SemVer 2.0.0 Specification.'
        assert not self.retired, f'Object already has been retired!'
        assert self.release, f'Can not retire a Object that has not been released!'

        assert self._release_version < retired_version, (
            f'Invalid retired version, it should be larger than the release version:\n'
            f'Release={self.release_version}\n'
            f'Retired={retired_version}'
        )
        self._retired = True
        self._retired_version = retired_version
        return

    def check(self) -> int:
        return self.status # This operation can check whether the status is valid.

    @property
    def status(self) -> int:
        status = (self._fresh_checker() << 2) + (self._release << 1) + (self._retired or self._legacy)
        # 0B100 -> It has never been added before.
        # 0B01X
        # 0B010 -> Release (New)
        # 0B011 -> Retired (Old)
        # 0B00X
        # 0B000 -> New
        # 0B001 -> Old
        assert 0B000 <= status and status <= 0B100, f'Invalid status code: {status}.'
        return status

    @property
    def is_fresh(self) -> bool:
        return self.status == 0B100

    @property
    def is_release(self) -> bool:
        return (self.status & 0B111) == 0B010

    @property
    def is_retired(self) -> bool:
        return (self.status & 0B111) == 0B011

    @property
    def is_internal(self) -> bool:
        return (self.status & 0B110) == 0B010

    @property
    def is_external(self) -> bool:
        return (self.status & 0B110) == 0B000

    @property
    def is_new(self) -> bool:
        return (self.status & 0B101) == 0B000

    @property
    def is_old(self) -> bool:
        return (self.status & 0B101) == 0B001

    def set_new(self) -> None:
        self._legacy = False
        return

    def set_old(self) -> None:
        self._legacy = True
        return