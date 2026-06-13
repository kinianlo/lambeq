# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Opt-in validation switch for the fast diagram core."""

from __future__ import annotations

import contextlib
import os

_enabled: bool = os.environ.get('LAMBEQ_FAST_VALIDATE', '') == '1'


def set_validation(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def validation_enabled() -> bool:
    return _enabled


@contextlib.contextmanager
def validation():
    global _enabled
    old, _enabled = _enabled, True
    try:
        yield
    finally:
        _enabled = old


@contextlib.contextmanager
def no_validation():
    global _enabled
    old, _enabled = _enabled, False
    try:
        yield
    finally:
        _enabled = old
