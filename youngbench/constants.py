#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-10 14:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import onnx


class Constant(object):
    def __setattr__(self, attribute_name, attribute_value):
        assert attribute_name not in self.__dict__, f'Constant Name exists: \"{attribute_name}\"'

        self.__dict__[attribute_name] = attribute_value

    def __contains__(self, attribute_value):
        return attribute_value in self._values_

    def freeze(self):
        values = set()
        for value in self.__dict__.values():
            if isinstance(value, set) or isinstance(value, frozenset):
                values = values.union(value)
            else:
                values.add(value)

        self._values_ = values


class YOUNG_BENCH_HANDLE(Constant):
    def initialize(self) -> None:
        handles = dict(
            Name = 'YoungBench',
            DatasetName = 'YB-Dataset',
            BenchmarkName = 'YB-Benchmark',
        )
        for name, value in handles.items():
            setattr(self, name, value)
        return


YoungBenchHandle = YOUNG_BENCH_HANDLE()
YoungBenchHandle.initialize()
YoungBenchHandle.freeze()


class ONNX_OPERATOR_DOMAIN(Constant):
    def initialize(self) -> None:
        domains = dict(
            DEFAULT = onnx.defs.ONNX_DOMAIN,
            ML = onnx.defs.ONNX_ML_DOMAIN,
            PREVIEW_TRAINING = onnx.defs.AI_ONNX_PREVIEW_TRAINING_DOMAIN,
        )
        for name, value in domains.items():
            setattr(self, name, value)
        return

ONNXOperatorDomain = ONNX_OPERATOR_DOMAIN()
ONNXOperatorDomain.initialize()
ONNXOperatorDomain.freeze()

class ONNX_OPERATOR_TYPE(Constant):
    def initialize(self) -> None:
        default_types = set()
        ml_types = set()
        preview_training_types = set()
        for op_schema in onnx.defs.get_all_schemas():
            if op_schema.domain == ONNXOperatorDomain.DEFAULT:
                default_types.add(op_schema.name)
            elif op_schema.domain == ONNXOperatorDomain.ML:
                ml_types.add(op_schema.name)
            elif op_schema.domain == ONNXOperatorDomain.PREVIEW_TRAINING:
                preview_training_types.add(op_schema.name)

        operator_types = dict(
            DEFAULT = frozenset(default_types),
            ML = frozenset(ml_types),
            PREVIEW_TRAINING = frozenset(preview_training_types),
        )
        for name, value in operator_types.items():
            setattr(self, name, value)
        return


ONNXOperatorType = ONNX_OPERATOR_TYPE()
ONNXOperatorType.initialize()
ONNXOperatorType.freeze()


class ONNX_OPERAND_TYPE(Constant):
    def initialize(self) -> None:
        operand_types = dict()
        for field in onnx.TypeProto.DESCRIPTOR.fields:
            if field.containing_oneof:
                operand_types[field.name] = field.number
        for name, value in operand_types.items():
            setattr(self, name, value)
        return


ONNXOperandType = ONNX_OPERAND_TYPE()
ONNXOperandType.initialize()
ONNXOperandType.freeze()


class ONNX_ELEMENT_TYPE(Constant):
    def initialize(self) -> None:
        element_types = dict()
        for element_type_name, element_type_number in onnx.TensorProto.DataType.items():
            element_types[element_type_name] = element_type_number
        for name, value in element_types.items():
            setattr(self, name, value)
        return


ONNXElementType = ONNX_ELEMENT_TYPE()
ONNXElementType.initialize()
ONNXElementType.freeze()


class ONNX_ATTRIBUTE_TYPE(Constant):
    def initialize(self) -> None:
        attribute_types = dict()
        for attribute_type_name, attribute_type_number in onnx.AttributeProto.AttributeType.items():
            attribute_types[attribute_type_name] = attribute_type_number
        for name, value in attribute_types.items():
            setattr(self, name, value)
        return


ONNXAttributeType = ONNX_ATTRIBUTE_TYPE()
ONNXAttributeType.initialize()
ONNXAttributeType.freeze()

ONNX = Constant()
ONNX.OPSetVersions = sorted(set(schema.since_version for schema in onnx.defs.get_all_schemas_with_history()))