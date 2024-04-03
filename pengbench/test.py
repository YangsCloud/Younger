from youngbench.dataset.modules.instance import Instance
import argparse
import pathlib
import onnx
import json

INSTANCE_DIR = "/Users/zrsion/instance-task/instances-Tasks-No-0/2en--HF--distilbert-base-uncased-finetuned-emotion--YBDI--model"

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=" ")
    # parser.add_argument('--instance-dir', type=pathlib.Path, default=None)
    # args = parser.parse_args()
    #print(INSTANCE_DIR)
    instance_obj = Instance()
    instance_obj.load(pathlib.Path(INSTANCE_DIR))
    instance_obj._load_network

    network = instance_obj.network
    graph = network.graph

    nodes = graph.nodes





    for node in nodes.data(data = 'attributes'):
        if not node[1]:
            continue
        for key in node[1]:
            if node[1][key]['attr_type']==onnx.defs.OpSchema.AttrType.INT or node[1][key]['attr_type']==onnx.defs.OpSchema.AttrType.INT:
                print(node[1][key]['attr_type'])


    # for op_type, attribute in zip(nodes(data = 'operator'),nodes(data = 'attributes')):
    #      print(op_type,attribute)




    #
    # print(nodes(data = 'attributes'))
    #
    # attributes = nodes(data = 'attributes')
    #
    # for attribute in attributes:
    #     print(attribute)


    # for node in nodes(data = 'operator'):
    #     print(node[1]['op_type'])



    # print(instance_obj.network.graph.adj)


