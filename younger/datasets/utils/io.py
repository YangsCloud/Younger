#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-14 14:55
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import sys
import onnx
import pathlib

from onnx import version_converter

from younger.commons.io import create_dir
from younger.commons.logging import logger

from younger.datasets.utils.constants import ONNX


def check_model(model_handler: onnx.ModelProto | pathlib.Path) -> bool:
    assert isinstance(model_handler, onnx.ModelProto) or isinstance(model_handler, pathlib.Path)
    # Change Due To Hash May Lead OOM.
    def check_with_internal() -> str | None:
        model = model_handler
        if len(model.graph.node) == 0:
            check_result = False
        else:
            onnx.checker.check_model(model)
            #check_result = hash_bytes(model)
            check_result = True
        return check_result

    def check_with_external() -> str | None:
        onnx.checker.check_model(str(model_handler))
        #model = onnx.load(str(model_handler))
        #check_result = hash_bytes(model.SerializeToString())
        check_result = True

        return check_result

    try:
        if isinstance(model_handler, onnx.ModelProto):
            return check_with_internal()
        if isinstance(model_handler, pathlib.Path):
            return check_with_external()
    except onnx.checker.ValidationError as check_error:
        logger.warn(f'The ONNX Model is invalid: {check_error}')
        check_result = False
    except Exception as error:
        logger.error(f'An error occurred while checking the ONNX model: {error}')
        sys.exit(1)
    return check_result


def load_model(model_filepath: pathlib.Path) -> onnx.ModelProto:
    model = onnx.load(model_filepath, load_external_data=False)
    return model


def save_model(model: onnx.ModelProto, model_filepath: pathlib.Path) -> None:
    create_dir(model_filepath.parent)
    onnx.save(model, model_filepath)
    return


def clean_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = version_converter.convert_version(model, ONNX.OPSetVersions[-1])
    return model