import onnx
import pathlib
import json
from younger.datasets.utils.constants import YoungerDatasetAddress, YoungerDatasetNodeType
from younger.commons.io import load_json, tar_extract

if __name__ == '__main__':

    save_dir = pathlib.Path("/Users/zrsion/YoungBench/pengbench")
    op_schemas = onnx.defs.get_all_schemas()


    op_type_list = list()
    for op_schema in op_schemas:
        op_type_list.append(op_schema.name)
        print(op_schema.name)

    with open(save_dir.joinpath('onnx_operators.json'), 'w') as f:
        json.dump(op_type_list, f, indent=4)

    # onnx_operators: list[str] = load_json("/Users/zrsion/YoungBench/pengbench/onnx_operators.json")
    # node_types = [f'__{node_type}__' for node_type in YoungerDatasetNodeType.attributes] + onnx_operators
    # node_type_indices = {node_type: index for index, node_type in enumerate(node_types)}
    # print(node_type_indices)


