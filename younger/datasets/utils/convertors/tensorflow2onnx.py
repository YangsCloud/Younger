#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-10 16:22
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import tensorflow

from typing import Literal

from tf2onnx import tf_loader, optimizer, utils
from tf2onnx.graph import ExternalTensorStorage
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.tf_utils import compress_graph_def

from younger.commons.io import get_path_size
from younger.commons.logging import logger


def main_export(model_path: pathlib.Path, output_path: pathlib.Path, model_type: Literal['saved_model', 'keras', 'tflite', 'tfjs'] = 'saved_model'):
    # [NOTE] The Code are modified based on the official tensorflow-onnx source codes. (https://github.com/onnx/tensorflow-onnx/blob/main/tf2onnx/convert.py [Method: main])
    assert model_type in {'saved_model', 'keras', 'tflite', 'tfjs'}

    model_name = model_path.name
    large_model = 2*1024*1024*1024 < get_path_size(model_path) 

    if model_type == 'saved_model':
        frozen_graph, inputs, outputs, initialized_tables, tensors_to_rename = tf_loader.from_saved_model(
            model_path, None, None, return_initialized_tables=True, return_tensors_to_rename=True
        )

    if model_type == 'keras':
        frozen_graph, inputs, outputs = tf_loader.from_keras(
            model_path, None, None
        )

    if model_type == 'tflite':
        frozen_graph, inputs, outputs = tf_loader.from_keras(
            model_path, None, None
        )

    with tensorflow.device("/cpu:0"):
        with tensorflow.Graph().as_default() as tensorflow_graph:
            if large_model:
                const_node_values = compress_graph_def(frozen_graph)
                external_tensor_storage = ExternalTensorStorage()
            if model_type not in {'tflite', 'tfjs'}:
                tensorflow.import_graph_def(frozen_graph, name='')
            graph = process_tf_graph(tensorflow_graph, const_node_values=const_node_values)
            onnx_graph = optimizer.optimize_graph(graph, catch_errors=True)
            model_proto = onnx_graph.make_model(f'converted from {model_name}', external_tensor_storage=external_tensor_storage)

    logger.info(f'Successfully converted TensorFlow model {model_path} to ONNX')

    if large_model:
        utils.save_onnx_zip(output_path, model_proto, external_tensor_storage)
        logger.info(f'Zipped ONNX model is saved at {output_path}. Unzip before opening in onnxruntime.')
    else:
        utils.save_protobuf(output_path, model_proto)
        logger.info(f'ONNX model is saved at {output_path}')