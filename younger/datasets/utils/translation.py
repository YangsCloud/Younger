#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-12-13 23:42
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import sys
import onnx
import networkx

from typing import Any, Callable
from functools import partial
from onnx.shape_inference import infer_shapes
from onnx.inliner import inline_local_functions


def get_all_attributes_of_operator(op_type: str, max_inclusive_version: int, domain: str ='') -> dict[str, tuple[int, str]] | None:
    # All attributes have default value only contain types - {<AttrType.FLOAT: 1>, <AttrType.INT: 2>, <AttrType.STRING: 3>, <AttrType.INTS: 7>, <AttrType.STRINGS: 8>}
    # Thus, we only stringize/destringize all values with - (str(value)/ast.literal_eval(str(value)))
    try:
        attributes = dict()
        schema = onnx.defs.get_schema(op_type, max_inclusive_version, domain=domain)
        for attribute_name, attribute_define in schema.attributes.items():
            attributes[attribute_name] = (attribute_define.type.value, str(onnx.helper.get_attribute_value(attribute_define.default_value)))
    except:
        attributes = None

    return attributes


def trans_string_string_entry_proto(string_string_entry_proto: onnx.StringStringEntryProto) -> dict:
    key: str = string_string_entry_proto.key
    value: str = string_string_entry_proto.value
    string_string_entry_proto_dict = dict(
        key = key,
        value = value
    )
    return string_string_entry_proto_dict

def trans_operator_set_id_proto(operator_set_id_proto: onnx.OperatorSetIdProto) -> dict:
    domain: str = operator_set_id_proto.domain
    version: int = operator_set_id_proto.version
    operator_set_id_proto_dict = dict(
        domain = domain,
        version = version
    )
    return operator_set_id_proto_dict


def trans_type_proto(type_proto: onnx.TypeProto) -> dict:
    # TODO: Add code for handling 'map_type,' 'opaque_type,' 'optional,' and 'sequence' to support classical ML operators.
    # NOTE: DNN-only implementations of ONNX MAY elect to not support non-tensor values as input and output to graphs and nodes.
    # NOTE: These types are needed to naturally support classical ML operators.
    # NOTE: DNN operators SHOULD restrict their input and output types to tensors.
    # The standard ONNX data types.
    #   message TypeProto {
    #     message Tensor {
    #       optional int32 elem_type = 1;
    #       optional TensorShapeProto shape = 2;
    #     }
    #     message Sequence { // repeated T
    #       optional TypeProto elem_type = 1;
    #     };
    #     message Map { // map<K,V>
    #       optional int32 key_type = 1; // This field MUST refer to an integral type ([U]INT{8|16|32|64}) or STRING
    #       optional TypeProto value_type = 2;
    #     };
    #     message Optional { // wrapper for Tensor, Sequence, or Map
    #       optional TypeProto elem_type = 1;
    #     };
    #     message SparseTensor {
    #       optional int32 elem_type = 1;
    #       optional TensorShapeProto shape = 2;
    #     }
    #   
    #     oneof value {
    #       Tensor tensor_type = 1;
    #       Sequence sequence_type = 4;
    #       Map map_type = 5;
    #       Optional optional_type = 9;
    #       SparseTensor sparse_tensor_type = 8;
    #     }
    #     optional string denotation = 6; // An optional denotation can be used to denote the whole type with a standard semantic description as to what is stored inside. Refer to https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition for pre-defined type denotations.
    #   }
    #   See more details at proto code where define the enum type 'TypeProto': https://github.com/postrational/onnx/blob/master/onnx/onnx.proto
    #   Map and Sequence of TypeProto are different with MapProto and SequenceProto
    #   For MapProto and SequenceProto see: https://github.com/onnx/onnx/blob/main/onnx/onnx-data.proto
    #   This file contains the proto definitions for MapProto and SequenceProto.
    #   These protos are used to represent the data structures of maps and sequence for use in test data or ModelProto.

    # This project only support for SparseTensorType and TensorType now.
    # TensorShapeProto
    #   message TensorShapeProto {
    #     message Dimension {
    #       oneof value {
    #         int64 dim_value = 1;
    #         string dim_param = 2;
    #       };
    #     };
    #     repeated Dimension dim = 1;
    #   }
    # See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#static-tensor-shapes

    denotation: str = type_proto.denotation

    type_names = {
        'tensor_type',
        'sparse_tensor_type',
        'sequence_type',
        'optional_type',
        'map_type',
    }

    for type_name in type_names:
        if type_proto.HasField(type_name):
            if type_name in {'tensor_type', 'sparse_tensor_type'}:
                # onnx.helper.make_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto
                # onnx.helper.make_sparse_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto
                type_field = getattr(type_proto, type_name)
                elem_type: int = type_field.elem_type
                shape: list[int | str] = list()
                for dim in type_field.shape.dim:
                    if dim.HasField('dim_param'):
                        shape.append(dim.dim_param)
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                type_proto_dict = dict(
                    elem_type = elem_type,
                    shape = shape,
                    shape_denotation = denotation,
                )
            elif type_name in {'sequence_type', 'optional_type'}:
                type_field = getattr(type_proto, type_name)
                elem_type: onnx.TypeProto = type_field.elem_type
                type_proto_dict = dict(
                    elem_type = trans_type_proto(elem_type),
                )
            elif type_name in {'map_type'}:
                type_field = getattr(type_proto, type_name)
                key_type: int = type_field.key_type
                value_type: onnx.TypeProto = type_field.value_type
                type_proto_dict = dict(
                    key_type = key_type,
                    value_type = trans_type_proto(value_type),
                )
            else:
                print(f'Support for handling other field ({type_name}) functionality is not yet available.')
                raise NotImplementedError
            break
        else:
            type_proto_dict = dict()
    return type_proto_dict


def trans_value_info_proto(value_info_proto: onnx.ValueInfoProto) -> dict:
    # ValueInfoProto:
    #   TypeProto:
    #     Map: key_type, value_type
    #     Opaque: domain, name
    #     Optional: elem_type
    #     Sequence: elem_type
    #     SparseTensor: elem_type, shape // -> It is SparseTensorTypeProto, not SpareTensorProto
    #     Tensor: elem_type, shape  // -> It is TensorTypeProto, not TensorProto
    #   message ValueInfoProto { // Defines information on value, including the name, the type, and the shape of the value.
    #     optional string name = 1;     // namespace Value
    #     optional TypeProto type = 2;
    #     optional string doc_string = 3;
    #   }
    # ValueInfoProto is mostly used in input, output, and value_info field of the onnx graph.
    # Inputs and Outputs: Each main (top-level) graph MUST define the names, types and shapes of its inputs and outputs, which are specified as 'value_info' structures.  The main graph inputs and outputs are required to have a shape, indicating the rank, even though the exact dimensions need not be specified.
    # See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs

    # There are 2 way to get ValueInfo with SparseTensorType or TensorType.
    # 1. Make ValueInfo from 'elem_type' and 'shape' directly.
    #   A. onnx.helper.make_tensor_value_info(name: str, elem_type: int, shape: Sequence[str | int | None] | None, doc_string: str = '', shape_denotation: List[str] | None = None) → ValueInfoProto.
    #   B. onnx.helper.make_sparse_tensor_value_info(name: str, elem_type: int, shape: Sequence[str | int | None] | None, doc_string: str = '', shape_denotation: List[str] | None = None) → ValueInfoProto.
    # 2. Make TypeProto from 'elem_type' and 'shape', then make ValueInfo from TypeProto.
    #   A. onnx.helper.make_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto
    #   --> onnx.helper.make_value_info(name: str, type_proto: TypeProto, doc_string: str = '') → ValueInfoProto
    #   B. onnx.helper.make_sparse_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto
    #   --> onnx.helper.make_value_info(name: str, type_proto: TypeProto, doc_string: str = '') → ValueInfoProto
    # This project adopts the 2nd way because it allows the transformation of more types of TypeProto in the future.
    name: str = value_info_proto.name
    doc_string: str = value_info_proto.doc_string
    type_proto = trans_type_proto(value_info_proto.type)
    value_info_proto_dict = dict(
        name = name,
        type_proto = type_proto,
        doc_string = doc_string,
    )
    return value_info_proto_dict


def trans_tensor_proto(tensor_proto: onnx.TensorProto, neglect_tensor_values: bool = True) -> dict:
    # Collect arguments that are used in onnx.helper.make_tensor(name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = False) → TensorProto.
    #   1. Two fields 'data_location' and 'external_data' are added to support for storing large tensor values.
    #      Where DataLocation is a new enum:
    #      enum DataLocation {
    #          MESSAGE = 0;
    #          RAW = 1;
    #          EXTERNAL = 2;
    #      }
    #      Later it is changed into
    #      enum DataLocation {
    #          DEFAULT = 0; // - DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
    #          EXTERNAL = 1; // - EXTERNAL - data stored in an external location as described by external_data field.
    #      }
    #   2. 'name,' 'data_type,' 'raw_data,' and 'data_location' are 'presence' fields.
    #   3. Field 'dims' has defualt value - []
    #   See more details at ONNX Issues: https://github.com/onnx/onnx/issues/5608 and https://github.com/onnx/onnx/pull/678
    #   See more details at proto code where define the enum type 'DataLocation': https://github.com/postrational/onnx/blob/master/onnx/onnx.proto
    # TODO: Add code for handling 'external_data'

    # TensorProto is mostly used in initializer field of the onnx graph.
    # Initializer can be the default value of an input or specifies a constant value.
    #   1. Defualt value of Input: When an initializer has the same name as a graph input, it specifies a default value for that input.
    #       When a name appears in both the initializer list and the graph input list, a runtime MAY allow a caller to specify a value for this (input) name overriding the value specified in the initializer and a runtime MAY allow users to omit specifying a value for this (input) name, choosing the value specified in the initializer.
    #   2. Constant Value: When an initializer has a name different from all graph inputs, it specifies a constant value.
    #       Names of constants that are not meant to be overridden by the caller should appear only in the initializer list and not in the graph input list.
    #   See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs & https://onnx.ai/onnx/repo-docs/IR.html#nodes

    name: str = tensor_proto.name
    data_type: int = tensor_proto.data_type
    dims: list[int] = list(tensor_proto.dims)

    if not neglect_tensor_values:
        # vals can be any type of data defined in onnx.TensorProto.DataType enum.
        # There is an easy alternative method:
        #   1. Make a TensorProto by using onnx.numpy_helper.from_array(arr: ndarray, name: str | None = None) → TensorProto
        #   2. Make a Numpy Arrray by using onnx.numpy_helper.to_array(tensor: TensorProto, base_dir: str = '') → ndarray
        if tensor_proto.data_location == onnx.TensorProto.DataLocation.DEFAULT:
            if tensor_proto.raw_data:
                raw: bool = True
                vals = tensor_proto.raw_data
            else:
                raw: bool = False
                field = onnx.helper.tensor_dtype_to_field(data_type)
                vals = list(getattr(tensor_proto, field))

        elif tensor_proto.data_location == onnx.TensorProto.DataLocation.EXTERNAL:
            print(f'Support for handling external_data functionality is not yet available.')
            raise NotImplementedError
        else:
            print(f'The ONNX proto file may have been updated, and the project does not yet support processing this type of data_location: {tensor_proto.data_location}.')
            raise NotImplementedError
    else:
        raw: bool = False
        vals = None

    tensor_proto_dict = dict(
        name = name,
        data_type = data_type,
        dims = dims,
        vals = vals,
        raw = raw,
    )
    return tensor_proto_dict


def trans_sparse_tensor_proto(sparse_tensor_proto: onnx.SparseTensorProto, neglect_tensor_values: bool = True) -> dict:
    # Collect arguments that are used in onnx.helper.make_sparse_tensor(values: TensorProto, indices: TensorProto, dims: Sequence[int]) → SparseTensorProto.
    # SparseTensorProto:
    #   The sequence of non-default values are encoded as a tensor of shape [NNZ] (the Number of NonZero elements).
    #   The default-value is zero for numeric tensors, and empty-string for string tensors.
    #   message SparseTensorProto {
    #     optional TensorProto values = 1;
    #     optional TensorProto indices = 2;
    #     repeated int64 dims = 3;
    #   }
    #   See more details at proto code where define the 'SparseTensorProto': https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx.proto
    #   See more details at ONNX Official Doc: https://onnx.ai/onnx/api/classes.html#sparsetensorproto

    # SparseTensorProto is mostly used in sparse_initializer field of the onnx graph.
    # values must have a non-empty name present which serves as a name for SparseTensorProto when used in sparse_initializer list.

    values = trans_tensor_proto(sparse_tensor_proto.values) if not neglect_tensor_values else None
    indices = trans_tensor_proto(sparse_tensor_proto.indices) if not neglect_tensor_values else None
    dims: list[int] = list(sparse_tensor_proto.dims)

    sparse_tensor_proto_dict = dict(
        name = values['name'],
        values = values,
        indices = indices,
        dims = dims
    )
    return sparse_tensor_proto_dict


def trans_attribute_proto(attribute_proto: onnx.AttributeProto, trans_graph_proto_method: Callable[[onnx.GraphProto, ], Any], neglect_tensor_values: bool = True) -> dict:
    # Collect arguments that are used in onnx.helper.make_attribute(key: str, value: Any, doc_string: str | None = None, attr_type: int | None = None) → AttributeProto
    # 1. A named attribute containing either singular float, integer, string, graph, and tensor values, or repeated float, integer, string, graph, and tensor values.
    # 2. An AttributeProto MUST contain the name field, and *only one* of the following content fields, effectively enforcing a C/C++ union equivalent.
    # 3. Node attributes are used to pass literal (static) values to operators.
    # 4. An attribute MUST have only one of the value-carrying properties.
    #    message AttributeProto {
    #      reserved 12, 16 to 19;
    #      reserved "v";
    #      enum AttributeType { // Note: this enum is structurally identical to the OpSchema::AttrType enum defined in schema.h.  If you rev one, you likely need to rev the other.
    #        UNDEFINED = 0;
    #        FLOAT = 1;
    #        INT = 2;
    #        STRING = 3;
    #        TENSOR = 4;
    #        GRAPH = 5;
    #        SPARSE_TENSOR = 11;
    #        TYPE_PROTO = 13;
    #
    #        FLOATS = 6;
    #        INTS = 7;
    #        STRINGS = 8;
    #        TENSORS = 9;
    #        GRAPHS = 10;
    #        SPARSE_TENSORS = 12;
    #        TYPE_PROTOS = 14;
    #      }
    #
    #      optional string name = 1;           // namespace Attribute, the name field MUST be present for this version of the IR.
    #      optional string ref_attr_name = 21; // This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
    #      optional string doc_string = 13; // A human-readable documentation for this attribute. Markdown is allowed.
    #
    #      # 1. For 0.0.1 versions of the IR, this field was not defined, and implementations needed to use has_field heuristics to determine which value field was in use.
    #      # 2. For IR_VERSION 0.0.2 or later, this field MUST be set and match the f|i|s|t|... field in use.  This change was made to accommodate proto3 implementations.
    #      #    This Project only support IR version > 0.0.2
    #      optional AttributeType type = 20;   // discriminator that indicates which field below is in use
    #
    #      # Exactly ONE of the following fields must be present for this version of the IR
    #      optional float f = 2;               // float
    #      optional int64 i = 3;               // int
    #      optional bytes s = 4;               // UTF-8 string
    #      optional TensorProto t = 5;         // tensor value
    #      optional GraphProto g = 6;          // graph
    #      optional SparseTensorProto sparse_tensor = 22;  // sparse tensor value
    #      optional TypeProto tp = 14;          // type proto
    #
    #      repeated float floats = 7;          // list of floats
    #      repeated int64 ints = 8;            // list of ints
    #      repeated bytes strings = 9;         // list of UTF-8 strings
    #      repeated TensorProto tensors = 10;  // list of tensors
    #      repeated GraphProto graphs = 11;    // list of graph
    #      repeated SparseTensorProto sparse_tensors = 23; // list of sparse tensors
    #      repeated TypeProto type_protos = 15;// list of type protos
    #    }
    # See more details at proto code where define the 'AttributeProto': https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
    # See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#attributes
    key: str = attribute_proto.name
    doc_string: str = attribute_proto.doc_string
    attr_type: int = attribute_proto.type

    single_types: set[int] = {
        onnx.defs.OpSchema.AttrType.FLOAT,
        onnx.defs.OpSchema.AttrType.INT,
        onnx.defs.OpSchema.AttrType.STRING,
        onnx.defs.OpSchema.AttrType.GRAPH,
        onnx.defs.OpSchema.AttrType.TENSOR,
        onnx.defs.OpSchema.AttrType.SPARSE_TENSOR,
        onnx.defs.OpSchema.AttrType.TYPE_PROTO,
    }

    repeat_types: set[int] = {
        onnx.defs.OpSchema.AttrType.FLOATS,
        onnx.defs.OpSchema.AttrType.INTS,
        onnx.defs.OpSchema.AttrType.STRINGS,
        onnx.defs.OpSchema.AttrType.GRAPHS,
        onnx.defs.OpSchema.AttrType.TENSORS,
        onnx.defs.OpSchema.AttrType.SPARSE_TENSORS,
        onnx.defs.OpSchema.AttrType.TYPE_PROTOS,
    }

    trans_method: dict[int, Callable[[Any]], Any] = {
        onnx.defs.OpSchema.AttrType.FLOAT: lambda x: x,
        onnx.defs.OpSchema.AttrType.FLOATS: lambda x: x,
        onnx.defs.OpSchema.AttrType.INT: lambda x: x,
        onnx.defs.OpSchema.AttrType.INTS: lambda x: x,
        onnx.defs.OpSchema.AttrType.STRING: lambda x: x,
        onnx.defs.OpSchema.AttrType.STRINGS: lambda x: x,

        onnx.defs.OpSchema.AttrType.GRAPH: trans_graph_proto_method,
        onnx.defs.OpSchema.AttrType.GRAPHS: trans_graph_proto_method,

        onnx.defs.OpSchema.AttrType.TENSOR: partial(trans_tensor_proto, neglect_tensor_values=neglect_tensor_values),
        onnx.defs.OpSchema.AttrType.TENSORS: partial(trans_tensor_proto, neglect_tensor_values=neglect_tensor_values),

        onnx.defs.OpSchema.AttrType.SPARSE_TENSOR: partial(trans_sparse_tensor_proto, neglect_tensor_values=neglect_tensor_values),
        onnx.defs.OpSchema.AttrType.SPARSE_TENSORS: partial(trans_sparse_tensor_proto, neglect_tensor_values=neglect_tensor_values),

        onnx.defs.OpSchema.AttrType.TYPE_PROTO: trans_type_proto,
        onnx.defs.OpSchema.AttrType.TYPE_PROTOS: trans_type_proto,
    }

    if attr_type in single_types:
        attribute_proto_value = onnx.helper.get_attribute_value(attribute_proto)
        value = trans_method[attr_type](attribute_proto_value)

    if attr_type in repeat_types:
        attribute_proto_values = onnx.helper.get_attribute_value(attribute_proto)
        value = list()
        for attribute_proto_value in attribute_proto_values:
            value.append(trans_method[attr_type](attribute_proto_value))

    attribute_proto_dict = dict(
        key = key,
        value = value,
        doc_string = doc_string,
        attr_type = attr_type
    )
    return attribute_proto_dict


def trans_node_io(node_proto: onnx.NodeProto) -> tuple[dict[str, int], dict[str, int]]:
    # Process inputs
    # An input of a node specifies a Tail endpoint of a Hidden edge.
    operands: dict[str, int] = dict()
    for node_input_index, node_input in enumerate(node_proto.input):
        if node_input == '':
            continue
        else:
            operands[node_input] = node_input_index

    # Process outputs
    # An output of a node specifies a Head endpoint of a Hidden edge.
    results: dict[str, int] = dict()
    for node_output_index, node_output in enumerate(node_proto.output):
        if node_output == '':
            raise onnx.defs.SchemaError
        else:
            results[node_output] = node_output_index

    return (operands, results)


def trans_node_proto(node_proto: onnx.NodeProto, opset_import: dict[str, int], trans_graph_proto_method: Callable[[onnx.GraphProto, ], Any], neglect_tensor_values: bool = True) -> dict:
    # TODO: Code ignores the processing of all third-party operators, excluding those defined by the official ONNX specification and user-defined functions.
    # TODO: In the future, functionality to handle these third-party operators will need to be added.
    # NOTE: A node input in a nested subgraph MAY refer to names introduced in outer graphs (as node outputs, graph inputs, or graph initializers).
    # NOTE: In the case of a nested subgraph, a node output name MUST be distinct from the names from the outer scopes that are visible in the nested subgraph.
    # Collect arguments that are used in onnx.helper.make_node(op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str | None = None, doc_string: str | None = None, domain: str | None = None, **kwargs: Any) → NodeProto
    # message NodeProto {
    #   repeated string input = 1;    // namespace Value
    #   repeated string output = 2;   // namespace Value
    #
    #   optional string name = 3;     // namespace Node, this field MAY be absent in some version of the IR
    #
    #   optional string op_type = 4;  // namespace Operator
    #   optional string domain = 7;   // namespace Domain
    #
    #   repeated AttributeProto attribute = 5; // Named attributes, another form of operator parameterization, used for constant values rather than propagated values.
    #
    #   optional string doc_string = 6;
    # }
    # See more details at proto code where define the 'NodeProto': https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
    # See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#nodes
    name: str = node_proto.name
    doc_string: str = node_proto.doc_string
    op_type: str = node_proto.op_type
    domain: str = node_proto.domain

    try:
        schema = onnx.defs.get_schema(op_type, max_inclusive_version=opset_import[domain], domain=domain)
    except onnx.defs.SchemaError:
        schema = None
    except Exception as exception:
        print(f'Caught an exception: {exception}')
        sys.exit(exception)

    # The option of the schema inputs can be: 'Single,' 'Optional,' or 'Variadic.'
    # 'Optional':
    #   1. Omit the last input/output;
    #   2. Provide an empty name of the optional input/output;
    #   3. MUST provide names for the calculated optional outputs & MUST NOT provide names of those not calculated.
    #   See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/ONNXTypes.html
    # 'Variadic':
    #   1. The last input or output of an operator MAY be marked as variadic.
    #   2. If the last input of the schema is 'Variadic,' the length of inputs can be larger than the schema's.
    #   See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#variadic-inputs-and-outputs

    if schema is None:
        pass
    else:
        # There are two ways to leave an optional input or output unspecified:
        # 1. The first, available only for trailing inputs and outputs, is to simply not provide that input;
        # 2. The second method is to use an empty string in place of an input or output name.

        # Check inputs
        # For each variadic operator input, N or more node inputs must be specified where N is the minimum arity of the operator.
        # Some operators have inputs that are marked as optional, which means that a referring node MAY forgo providing values for such inputs.
        if len(schema.inputs) > 0:
            last_input = schema.inputs[-1]
            if last_input.option.value == last_input.option.Optional.value:
                assert len(node_proto.input) <= len(schema.inputs)
            if last_input.option.value == last_input.option.Variadic.value:
                assert len(node_proto.input) >= len(schema.inputs) - 1

        # Check outputs
        # For each variadic operator output, N or more node outputs must be specified where N is the minimum arity of the operator.
        # Some operators have outputs that are optional. When an actual output parameter of an operator is not specified, the operator implementation MAY forgo computing values for such outputs.
        # Each node referring to an operator with optional outputs MUST provide a name for each output that is computed and MUST NOT provide names for outputs that are not computed.
        if len(schema.outputs) > 0:
            last_output = schema.outputs[-1]
            if last_output.option.value == last_output.option.Optional.value:
                assert len(node_proto.output) <= len(schema.outputs)
            if last_output.option.value == last_output.option.Variadic.value:
                assert len(node_proto.input) >= len(schema.inputs) - 1

    (operands, results) = trans_node_io(node_proto)

    # Process attributes
    # Record whether there is a subgraph
    node_proto_attributes: dict[str, dict] = dict()
    for node_proto_attribute in node_proto.attribute:
        node_proto_attribute = trans_attribute_proto(node_proto_attribute, trans_graph_proto_method=trans_graph_proto_method, neglect_tensor_values=neglect_tensor_values)
        key = node_proto_attribute.pop('key')
        node_proto_attributes[key] = node_proto_attribute

    node_proto_dict = dict(
        name = name,
        doc_string = doc_string,
        operator = dict(op_type=op_type, domain=domain),
        operands = operands,
        results = results,
        attributes = node_proto_attributes
    )
    return node_proto_dict


def trans_graph_proto(ox_graph: onnx.GraphProto, opset_import: dict[str, int], outer_dataflow2source: dict[str, tuple[int, int]] | None = None, neglect_tensor_values: bool = True, verbose: bool = False) -> networkx.DiGraph:
    # TODO: Some `value_info` may not be inferred by the `onnx.shape_inference.infer_shapes` method.
    # TODO: This is because there are operators outside the official ONNX domain in the graph.
    # TODO: Support for this part will need to be added in the future.
    outer_dataflow2source = outer_dataflow2source or dict()
    nx_graph = networkx.DiGraph()

    def get_complete_node_attributes(node_type: str, node_attributes: dict = None) -> dict:
        assert node_type in {'outer', 'input', 'output', 'constant', 'operator'}
        node_attributes = node_attributes or dict()

        outer_attributes = dict(
            outer_names = None,
        )
        input_attributes = dict(
            graph_inputs = None,
        )
        output_attributes = dict(
            graph_outputs = None,
        )
        constant_attributes = dict(
            graph_constants = None,
        )
        operator_attributes = dict(
            name = None,
            doc_string = None,
            operator = None,
            operands = None,
            results = None,
            attributes = None,
        )

        if node_type == 'outer':
            specific_attributes = outer_attributes
        if node_type == 'input':
            specific_attributes = input_attributes
        if node_type == 'output':
            specific_attributes = output_attributes
        if node_type == 'constant':
            specific_attributes = constant_attributes
        if node_type == 'operator':
            specific_attributes = operator_attributes

        for node_attribute_key, node_attribute_value in node_attributes.items():
            if node_attribute_key in specific_attributes.keys():
                specific_attributes[node_attribute_key] = node_attribute_value

        complete_node_attributes = dict(type=node_type)
        complete_node_attributes.update(outer_attributes)
        complete_node_attributes.update(input_attributes)
        complete_node_attributes.update(output_attributes)
        complete_node_attributes.update(constant_attributes)
        complete_node_attributes.update(operator_attributes)

        return complete_node_attributes
    # A onnx graph defines the computational logic of a model and is comprised of a parameterized list of nodes that form a directed acyclic graph based on their inputs and outputs.
    # This is the equivalent of the "network" or "graph" in many deep learning frameworks.
    # message GraphProto {
    #   repeated NodeProto node = 1; // -> processed in this method
    #   optional string name = 2;
    #   repeated TensorProto initializer = 5; // -> see trans_tensor_proto_from_ox_to_nx
    #   repeated SparseTensorProto sparse_initializer = 15; // -> see trans_sparse_tensor_proto_from_ox_to_nx

    #   optional string doc_string = 10;
    #   repeated ValueInfoProto input = 11; // -> see trans_value_info_proto_from_ox_to_nx
    #   repeated ValueInfoProto output = 12; // -> see trans_value_info_proto_from_ox_to_nx
    #
    #   repeated ValueInfoProto value_info = 13; // -> see trans_value_info_proto_from_ox_to_nx
    #
    #   repeated TensorAnnotation quantization_annotation = 14;
    #
    #   reserved 3, 4, 6 to 9;
    #   reserved "ir_version", "producer_version", "producer_tag", "domain";
    # }
    #  'quantization_annotation' field carries information to indicate the mapping among a tensor and its quantization parameter tensors.
    #  For example:
    #    For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated, which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.

    # All nodes of nx_graph contains 4 types, the type of node are specified by the node attribute: 'type'
    #   1. 'operator' nodes (>=1), which are recorded in 'node' field of ox_graph;
    #     -. type='operator'
    #     -. other attributes are unpack from dict 'nx_graph_node' = trans_node_proto_from_ox_to_nx(ox_graph_node)
    #       --. 'name,' 'doc_string,' 'operator,' 'operands,'(Dict[node_input_name, node_input_index]) 'results,'(Dict[node_output_name, node_output_index]) and 'attributes.'
    #   2. 'input' node (=1), which can be constructed from 'input' field of ox_graph;
    #     -. type='input'
    #     -. only contains 1 additional attribute, 'graph_inputs'(Dict[input_value_info_name, input_value_info_index])
    #   3. 'output' node (=1), which can be constructed from 'output' field of ox_graph;
    #     -. type='output'
    #     -. only contains 1 additional attribute, 'graph_outputs'(Dict[output_value_info_name, output_value_info_index])
    #   4. 'constant' node (=1 or =0), which can be constructed from 'input' and 'initializer' field of ox_graph;
    #     -. type='constant'
    #     -. only contains 1 additional attribute, 'graph_constants'(Dict[constant_tensor_name, constant_tensor_index])
    #       --. constant_tensor belongs to the relative complement of input_value_info with respect to a set initializer (initializer / input_value_info)
    # All edges of nx_graph contains 4 types, but there is no need to explicitly distinguish between them:
    #   An edge start from a node which label is 'tail_index' and end to a node which label is 'head_index';
    #   They all have 3 attributes: 'connection'=dict('trap_index'=int, 'emit_index'=int) , 'dataflow,' and 'default_value'
    #     a. 'head_index' records the index of the node to which the data flow is input.
    #     b. 'trap_index' records which occurrence of "operand" in the node specified by 'head_index' received the data.
    #     c. 'tail_index' records the index of the node from which the data flow output.
    #     d. 'emit_index' records which occurrence of 'result' in the node specified by 'tail_index' emitted the data.
    #     e. 'dataflow'  records the value_info of the graph (include 'input,' 'output,' and 'value_info' field).
    #     f. 'default_value' records the tensors in 'initializer' field of graph.
    #   1. node to node, recorded in 'value_info';
    #     -. 'default_value' attribute is None;
    #   2. input to node, recorded in 'input' field of ox_graph;
    #     -. 'default_value' attribute MAY NOT be None (it may be recorded in 'initializer' field as default value of the input);
    #   3. constant to node, recorded in 'initializer' field of ox_graph;
    #     -. 'default_value' attribute MUST be set (it is recorded in 'initializer' field);
    #     -. 'dataflow' attribute is set to be None. To our knowledge, there is no direct evidence proving that a constant MUST have a value_info.
    #   4. node to output, recorded in 'output' field of ox_graph;
    #     -. 'default_value' attribute MUST be None;

    # This Project: Dataflow
    #   1. The output of a node can be sent as input to other nodes, and it may even serve as several different inputs for another node.
    #   2. This project call the value send from one node to another node as dataflow (which is an edge of the nx_graph);
    #   3. So, some dataflow that have the same tail endpoint have the same name of value_info;
    #
    # # The occurrence of a name as a node output is said to be a definition (also graph input and graph initializer);
    # # The occurrence of a name as a node input is said to be a use (also graph output);
    #   -> So, the name of a node output corresponds a unique node;
    #   -> So, the name of a node input corresponds multiple nodes;
    #   This project uniquely identifies each dataflow using the node index of endpoints and the index of its input of head and its output of tail.
    # Within a namespace, each name MUST be unique for each given graph.
    # Value namespace contains: node inputs & outputs, tensor values (if named), graph inputs, outputs.
    # For the infomation about relationship between 'input,' 'output,' 'value_info,' and 'initializer', see ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs & https://onnx.ai/onnx/repo-docs/IR.html#names-within-a-graph

    # Process Dataflow
    dataflows: dict[str, dict] = dict() # Key: Name in Value namespace; Value: node inputs & outputs, tensor values (if named, tensor_type_proto), graph inputs, outputs. - All is ValueInfo
    default_values: dict[str, dict] = dict() # Key: Name in Value namespace; Value: tensor of inputs or constants. - All is Tensor

    graph_inputs: dict[str, int] = dict()
    graph_outputs: dict[str, int] = dict()
    graph_constants: dict[str, int] = dict()

    for input_value_info in ox_graph.input:
        input_value_info = trans_value_info_proto(input_value_info)
        name = input_value_info.pop('name')
        dataflows[name] = input_value_info
        graph_inputs[name] = len(graph_inputs)

    for output_value_info in ox_graph.output:
        output_value_info = trans_value_info_proto(output_value_info)
        name = output_value_info.pop('name')
        dataflows[name] = output_value_info
        graph_outputs[name] = len(graph_outputs)

    for internode_value_info in ox_graph.value_info:
        internode_value_info = trans_value_info_proto(internode_value_info)
        name = internode_value_info.pop('name')
        dataflows[name] = internode_value_info

    # Process initializer and sparse_initializer
    # Initializer can be the default value of an input or specifies a constant value.
    #   1. Defualt value of Input: When an initializer has the same name as a graph input, it specifies a default value for that input.
    #       When a name appears in both the initializer list and the graph input list, a runtime MAY allow a caller to specify a value for this (input) name overriding the value specified in the initializer and a runtime MAY allow users to omit specifying a value for this (input) name, choosing the value specified in the initializer.
    #   2. Constant Value: When an initializer has a name different from all graph inputs, it specifies a constant value.
    #       Names of constants that are not meant to be overridden by the caller should appear only in the initializer list and not in the graph input list.
    #   See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs & https://onnx.ai/onnx/repo-docs/IR.html#nodes

    non_constants = set() # initializer is not belong to graph inputs & node outputs (dataflow definitions)
    for input_value_info in ox_graph.input:
        non_constants.add(input_value_info.name)

    for node in ox_graph.node:
        for node_output in node.output:
            non_constants.add(node_output)

    for graph_initializer in ox_graph.initializer:
        graph_initializer = trans_tensor_proto(graph_initializer, neglect_tensor_values=neglect_tensor_values)
        name = graph_initializer.pop('name')
        default_values[name] = graph_initializer
        if name not in non_constants:
            graph_constants[name] = len(graph_constants)

    for graph_sparse_initializer in ox_graph.sparse_initializer:
        graph_sparse_initializer = trans_sparse_tensor_proto(graph_sparse_initializer, neglect_tensor_values=neglect_tensor_values)
        name = graph_sparse_initializer.pop('name')
        default_values[name] = graph_sparse_initializer
        if name not in non_constants:
            graph_constants[name] = len(graph_constants)

    # all ox_graph.input are saved in input node
    # all ox_graph.output are saved in output node
    # operator node index of nx_graph = index of ox_graph.node
    # input node index of nx_graph = len(ox_graph.node) + 0
    # output node index of nx_graph = len(ox_graph.node) + 1
    # constant node index of nx_graph = len(ox_graph.node) + 2
    # outer node index of nx_graph = len(ox_graph.node) + 3

    def check_official_io(io_name: str):
        if verbose and io_name not in dataflows:
            print(
                f'Please check the onnx model, \"{io_name}\" is certain to appear in the \"dataflow\", it MAY be an unofficial ONNX Operator.'
                f'This MAY be because there are operators outside the official ONNX domain in the graph.'
                f'OR it is an input_name that in a nested subgraph which refer to names introduced in outer graphs'
            )

    input_node_index = f'{len(ox_graph.node) + 0}'
    output_node_index = f'{len(ox_graph.node) + 1}'
    constant_node_index = f'{len(ox_graph.node) + 2}'
    outer_node_index = f'{len(ox_graph.node) + 3}'

    # Map names introduced in graphs to its tail_index node and emit_index output.
    dataflow2source: dict[str, tuple[int, int]] = dict()
    for node_index, node in enumerate(ox_graph.node):
        node_index = f'{node_index}'
        (_, results) = trans_node_io(node)
        for result_name, result_index in results.items():
            dataflow2source[result_name] = (node_index, result_index)
            check_official_io(result_name)

    for input_name, input_index in graph_inputs.items():
        dataflow2source[input_name] = (input_node_index, input_index)
        check_official_io(input_name)

    for constant_name, constant_index in graph_constants.items():
        dataflow2source[constant_name] = (constant_node_index, constant_index)
        check_official_io(constant_name)

    # Add nx_graph nodes
    for node in ox_graph.node:
        all_outer_dataflow2source = dict()
        all_outer_dataflow2source.update(dataflow2source)
        all_outer_dataflow2source.update(outer_dataflow2source)
        trans_graph_proto_method = partial(trans_graph_proto, opset_import=opset_import, outer_dataflow2source=all_outer_dataflow2source, neglect_tensor_values=neglect_tensor_values, verbose=verbose)
        node = trans_node_proto(node, opset_import, trans_graph_proto_method, neglect_tensor_values=neglect_tensor_values)
        nx_graph.add_node(f'{len(nx_graph)}', **get_complete_node_attributes('operator', node))

    nx_graph.add_node(input_node_index, **get_complete_node_attributes('input', dict(graph_inputs=graph_inputs)))
    nx_graph.add_node(output_node_index, **get_complete_node_attributes('output', dict(graph_outputs=graph_outputs)))
    nx_graph.add_node(constant_node_index, **get_complete_node_attributes('constant', dict(graph_constants=graph_constants)))
    nx_graph.add_node(outer_node_index, **get_complete_node_attributes('outer'))

    # Add nx_graph edges
    # TODO: Multiple (tail_index, head_index)? Typically Not Encounter?
    for node_index, node in nx_graph.nodes.items():
        if node['operands'] is not None:
            inputs_key = 'operands'
        elif node['graph_outputs'] is not None:
            inputs_key = 'graph_outputs'
        else:
            continue
        for input_name, input_index in node[inputs_key].items():
            check_official_io(input_name)
            (head_index, trap_index) = (node_index, input_index)
            if input_name in dataflow2source:
                (tail_index, emit_index) = dataflow2source[input_name]
                attributes = dict(
                    connection = dict(emit_index=emit_index, trap_index=trap_index),
                    default_value = default_values.get(input_name, None),
                    dataflow = dataflows.get(input_name, None),
                )
            elif input_name in outer_dataflow2source:
                tail_index = outer_node_index
                attributes = dict(
                    connection = dict(emit_index=None, trap_index=trap_index),
                    default_value = None,
                    dataflow = None,
                )
            else:
                print(f'{input_name} not defined!')
                raise KeyError
            nx_graph.add_edge(tail_index, head_index, **attributes)

    return nx_graph


def trans_model_proto(model: onnx.ModelProto, neglect_tensor_values: bool = True, verbose: bool = False) -> networkx.DiGraph:
    # Tranlating components of ModelProto into DiGraph attributes.
    # It is a detailed version of the 'trans_model_proto'
    # ModelProto provides more metadata than GraphProto

    # The main purpose of the model structure is to associate metadata with a graph which contains all the executable elements.
    # The metadata is used when first reading the model file, giving an implementation the information it needs in order to determine whether it will be able to execute the model, generate logging messages, error reports, etc.
    # Further, the metadata is useful to tools, such as IDEs and model galleries, which need it for informing humans about a given model’s purpose and characteristics.

    # message ModelProto {
    #   optional int64 ir_version = 1; // The version of the IR this model targets. See Version enum above. This field MUST be present.
    #   repeated OperatorSetIdProto opset_import = 8; // All ModelProtos MUST have at least one entry that specifies which version of the ONNX OperatorSet is being imported.
    #
    #   optional string producer_name = 2; // The name of the framework or tool used to generate this model. This field SHOULD be present to indicate which implementation/tool/framework emitted the model.
    #   optional string producer_version = 3; // The version of the framework or tool used to generate this model. This field SHOULD be present to indicate which implementation/tool/framework emitted the model.
    #
    #   optional string domain = 4; // Together with `model_version` and GraphProto.name, this forms the unique identity of the graph. For example: `com.facebook.fair` or `com.microsoft.cognitiveservices`
    #   optional int64 model_version = 5;
    #   optional string doc_string = 6;
    #   optional GraphProto graph = 7;
    #   repeated StringStringEntryProto metadata_props = 14; // Named metadata values; keys should be distinct.
    #   repeated TrainingInfoProto training_info = 20;
    #
    #   repeated FunctionProto functions = 25;
    # };
    assert isinstance(model, onnx.ModelProto), f'Argument \"model\" must be an ONNX Model Proto (onnx.ModelProto) instead \"{type(model)}\"!'

    # If IR version >= 3, the model must specify opset_import. If IR version < 3, the model cannot have any opset_import specified.
    # https://onnx.ai/onnx/api/checker.html#onnx.checker.check_model
    # https://onnx.ai/onnx/repo-docs/Versioning.html#released-versions
    assert 3 <= model.ir_version, f'IR Version {model.ir_version} Not Support! Only Accept 3 <= IR Version (1.0 <= ONNX Version).'

    model = inline_local_functions(model) # Expand all local functions of the model.
    model = infer_shapes(model) # Infer all shape of hiddens.

    ir_version: str = model.ir_version

    producer_name: str = model.producer_name
    producer_version: str = model.producer_version
    domain: str = model.domain
    model_version: int = model.model_version
    doc_string: str = model.doc_string

    # OperatorSetIdProto
    # This is the type of attribute opset_import of class ModelProto.
    # This attribute specifies the versions of operators used in the model.
    # Every operator or node belongs to a domain. All operators for the same domain share the same version.
    # https://onnx.ai/onnx/api/classes.html#operatorsetidproto
    # It's schema can be loaded by using onnx.defs.get_schema(*args, **kwargs)
    # 1. get_schema(op_type: str, max_inclusive_version: int, domain: str = ‘’) -> onnx.onnx_cpp2py_export.defs.OpSchema
    #    Return the schema of the operator op_type and for a specific version.
    # 2. get_schema(op_type: str, domain: str = ‘’) -> onnx.onnx_cpp2py_export.defs.OpSchema
    #    Return the schema of the operator op_type and for a specific version.
    # https://onnx.ai/onnx/api/defs.html#onnx.defs.get_schema
    opset_import: dict[str, int] = dict()
    for ox_model_opset_import in model.opset_import:
        ox_model_opset_import: dict[str, str | int] = trans_operator_set_id_proto(ox_model_opset_import)
        domain: str = ox_model_opset_import['domain']
        version: int = ox_model_opset_import['version']
        opset_import[domain] = version

    metadata_props: list[dict[str, str]] = list()
    for ox_model_metadata_props in model.metadata_props:
        ox_model_metadata_props: dict[str, str] = trans_string_string_entry_proto(ox_model_metadata_props)
        metadata_props.append(ox_model_metadata_props)

    graph: networkx.DiGraph = trans_graph_proto(model.graph, opset_import=opset_import, neglect_tensor_values=neglect_tensor_values, verbose=verbose)

    graph_attributes = dict(
        ir_version = ir_version,
        opset_import = opset_import,
        producer_name = producer_name,
        producer_version = producer_version,
        domain = domain,
        model_version = model_version,
        doc_string = doc_string,
        metadata_props = metadata_props,
    )

    graph.graph.update(**graph_attributes)

    return graph
