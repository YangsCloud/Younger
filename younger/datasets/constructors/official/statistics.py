#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Luzhou Peng (彭路洲) and Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-17 17:20
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
from younger.datasets.modules import Instance
import pathlib
import networkx
from typing import Dict 
import itertools


instance_dir = pathlib.Path('/Users/zrsion/instances-0.0.1') # test
save_dir = pathlib.Path('/Users/zrsion/instances_statistics') # test

            
def get_operators(nodes: networkx.DiGraph.nodes) -> Dict[str, int]:
    operators_dict = dict()
    for node in nodes(data = 'operator'):
        if not node[1]: # node : e.g. ('589', None), node[1]==None , This will skip nodes with node_type == 'input' | 'output' | 'constant' | 'outer'
            continue
            
        operator = str(node[1])
        if operator not in operators_dict:
            operators_dict[operator] = 1
        else:
            operators_dict[operator] += 1
        
    return operators_dict


def generater_statistics(instance_dir: pathlib.Path, save_dir: pathlib.Path,
                         tasks: list[str], datasets: list[str], splits: list[str], metrics: list[str]):
    
    combinations = list()
    task_iter = tasks if tasks else ['']
    dataset_iter = datasets if datasets else ['']
    split_iter = splits if splits else ['']
    metric_iter = metrics if metrics else ['']

    for task, dataset, split, metric in itertools.product(task_iter, dataset_iter, split_iter, metric_iter):
        combination = (task,) + (dataset,) + (split,) + (metric,)
        combination = tuple(item for item in combination if item)
        combinations.append(str(combination))
    
    final_dic = dict()
    cnt = 0 # test
    
    instance_paths = list(instance_dir.iterdir())
    for instance_path in tqdm(instance_paths):
        instance = Instance()
        instance.load(instance_path)
        nodes = instance.network.graph.nodes
        model_id = str(instance_path).rsplit("/", 1)[-1]
        model_operators = get_operators(nodes)
        
        if cnt > 15: break # test
        
        if instance.labels:
            labels = instance.labels['labels']
            
            if labels:
                for label in labels:
                    task = label['task'] if tasks else ''
                    dataset = label['dataset'][0] if datasets else ''
                    split = label['dataset'][1] if splits else ''
                    metric = label['metric'][0] if metrics else ''
                    value = label['metric'][1]
                    
                    combination = (task,) + (dataset,) + (split,) + (metric,)
                    combination = str(tuple(item for item in combination if item))
                    
                    if combination in combinations:
                        dic_to_add = dict()
                        dic_to_add[model_id] = list()
                        dic_to_add[model_id].extend([value, model_operators])
                        if combination not in final_dic:
                            final_dic[combination] = list()
                        final_dic[combination].append(dic_to_add)
                        cnt += 1 # test
                        break
    
    save_file_name_suffix = "sta"
    if tasks:
        save_file_name_suffix += "_" + "tasks" 
    if datasets:
        save_file_name_suffix += "_" + "datasets" 
    if splits:
        save_file_name_suffix += "_" + "splits"
    if metrics:
        save_file_name_suffix += "_" + "metrics" 
    save_file_name_suffix = save_file_name_suffix.strip("_") + ".json"
    
    with open(save_dir.joinpath(save_file_name_suffix), 'w') as f:
        json.dump(final_dic, f, indent=4)
    
                
if __name__ == '__main__':

    generater_statistics(instance_dir, save_dir, tasks=[], 
                         datasets=[], splits=["test", "train"], metrics=["accuracy","f1","wer"])
    
    # generater_statistics(instance_dir, save_dir, tasks=[], 
    #                      datasets=[], splits=[], metrics=[])

