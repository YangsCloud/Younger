import os

import onnx.defs

from youngbench.dataset.modules.instance import Instance
import argparse
import pathlib
from pathlib import Path
import networkx
from typing import Set, List, Dict, Optional
import json

op_type_dic = {}

final_dic = {
    "operator": {
        "type": "str",
        "total": "int",
        "attributes-types": {
            'attributes-1': {
                'value': {
                    'value': {
                        'name': '',
                        'data_type': 7,
                        'dims': [],
                        'vals': None,
                        'raw': False
                    },
                    'doc_string': '',
                    'attr_type': 4,
                    "total": "int"
                }
            }
        }
    }
}


def sta_op_type(nodes: networkx.DiGraph.nodes):
    for node in nodes(data='operator'):
        if not node[1] or not 'op_type' in node[1]:  # node : e.g. ('589', None), node[1]==None
            op_type = str(node[1])
            # print(op_type)

        # print(node[1]['op_type'])
        else:
            op_type = node[1]['op_type']
        if op_type in op_type_dic:
            op_type_dic[op_type] += 1
            # value = op_type_dic[op_type]
            # print(f"{op_type} : {value}")
        else:
            op_type_dic[op_type] = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")

    parser.add_argument('--instance-task-dir', type=pathlib.Path, default=None)
    parser.add_argument('--save-dir', type=pathlib.Path, default=None)

    args = parser.parse_args()

    instance_task_dir = pathlib.Path(args.instance_task_dir)
    save_dir = pathlib.Path(args.save_dir)
    nodes_info_path = save_dir.joinpath('nodes_info.txt')

    nodes_info_data = ""

    for instance_dir in os.listdir(instance_task_dir):
        if not instance_dir.endswith('.flg') and not instance_dir.endswith('.tar') and not instance_dir.endswith('.DS_Store') and not instance_dir.endswith('json') and not instance_dir.endswith('.txt') and not instance_dir.endswith('TEST'):
            instance_dir = pathlib.Path(instance_task_dir.joinpath(instance_dir))
            print(f"now processing {instance_dir}")
            # print(instance_dir)
            #
            # cnt += 1
            # if cnt > 25:
            #     continue
            # print(instance_dir)

            for file in os.listdir(instance_dir):
                if file.endswith('.DS_Store') or file.endswith('.txt'):
                    continue
                instance_path = instance_dir.joinpath(file)
                instance_obj = Instance()
                instance_obj.load(instance_path)
                network = instance_obj.network
                graph = network.graph
                nodes = graph.nodes

                for node in nodes.data(data='attributes'):
                    if not node[1]:
                        # print("pass")
                        continue
                    for key in node[1]:
                        # print(node[1][key]['attr_type'])
                        if node[1][key]['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPH or node[1][key]['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPHS:
                            print(f'has graph, the instance_path is {instance_path}')






