from base64 import encode
from importlib.resources import path
import re
import torch
import pathlib
import argparse
import ast
import json
import numpy as np
import tqdm
import torch
import torch.utils.data
import multiprocessing
import matplotlib.pyplot as plt

from typing import Any, Literal, List

from younger.applications.utils.neural_network import get_model_parameters_number, get_device_descriptor, fix_random_procedure, set_deterministic, load_checkpoint, save_checkpoint

from sklearn.manifold import TSNE

from younger.commons.logging import Logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.datasets.node_dataset import NodeDataset
from younger.applications.models import GCN_NP, GAT_NP, SAGE_NP
from younger.applications.datasets import BlockDataset
from younger.applications.tasks.base_task import YoungerTask

def get_model(baseline_model: str, encode_type: str): 
    if baseline_model == 'gcn':
        model = GCN_NP(
            node_dict_size=185 if encode_type == 'operator' else 4409,
            node_dim=1024,
            hidden_dim=512,
            dropout=0.5,
            output_embedding=True,
        )
    elif baseline_model == 'gat':
        model = GAT_NP(
            node_dict_size=185 if encode_type == 'operator' else 4409,
            node_dim=1024,
            hidden_dim=512,
            dropout=0.5,
            output_embedding=True,
        )
    elif baseline_model == 'sage':
        model = SAGE_NP(
            node_dict_size=185 if encode_type == 'operator' else 4409,
            node_dim=1024,
            hidden_dim=512,
            dropout=0.5,
            output_embedding=True,
        )
    return model

def draw_tsne(datas: List[np.ndarray],save_dir, save_name, mark):
    numpy_array = datas
    y = np.array(mark)
    print(numpy_array.shape)
    print(y.shape)
    tab20_cmap = plt.get_cmap('tab20')
    colors = tab20_cmap.colors
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(numpy_array)

    plt.figure(figsize=(14, 14))
    plt.scatter(embedding_2d[y == 0, 0], embedding_2d[y == 0, 1], color=colors[1], label='Other operators')
    plt.scatter(embedding_2d[y == 1, 0], embedding_2d[y == 1, 1], color=colors[2], label='Selected operators')

    plt.title('t-SNE visualization', fontsize=29)
    plt.tick_params(axis='both', labelsize=28)
    plt.xlabel('Dimension 1', fontsize=28)
    plt.ylabel('Dimension 2', fontsize=28)
    plt.legend(fontsize=20,scatterpoints=3)
    
    plt.savefig(save_dir.joinpath(f'{save_name}_node_embedding.pdf'),format='pdf')
    plt.savefig(save_dir.joinpath(f'{save_name}_node_embedding.jpg'),format='jpg')


def draw_node_emb(mark, save_dir, embedding_model, encode_type, save_name, checkpoint_file):
    model = get_model(embedding_model, encode_type)
    model.eval()
    embedding_weights_before = model.node_embedding_layer.weight.data.cpu().numpy()
    draw_tsne(embedding_weights_before, save_dir, f'{save_name}_before', mark)

    model.eval()
    checkpoint = load_checkpoint(checkpoint_file)
    print('now loading checkpoint: ', checkpoint_file.stem)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    embedding_weights_after = model.node_embedding_layer.weight.data.cpu().numpy()
    draw_tsne(embedding_weights_after, save_dir, f'{save_name}_after', mark)


def node_embedding_process(dataset_dir: pathlib.Path, save_dir, embedding_model, encode_type, save_name, checkpoint_file):
    dataset = NodeDataset(
        root=dataset_dir.joinpath(encode_type, 'train'),
        worker_number=32,
        encode_type=encode_type,
    )
    print('root_dir:', dataset_dir.joinpath(encode_type, 'train'))
    meta = dataset.meta
    x_dict = dataset.x_dict
    map_dict = dict()
    if encode_type == 'node':
        operators = meta['all_nodes']['onnx']
    else:
        operators = meta['all_operators']['onnx']
    
    operators = sorted(operators, key=operators.get, reverse=True)
    sorted_keys_list = list(operators)

    top_list = sorted_keys_list[:500] if encode_type == 'node' else sorted_keys_list[:20]
    if encode_type == 'node':
        for key, value in x_dict['n2i'].items():
            if key == '__UNK__' or key == '__MASK__':
                map_dict[x_dict['n2i'][key]] = 0
                continue 

            if key in top_list:
                map_dict[x_dict['n2i'][key]] = 1
            else:
                map_dict[x_dict['n2i'][key]] = 0
        mark = [0] * len(map_dict)
        for key, value in map_dict.items():
            mark[key] = value
    
    else:
        for key, value in x_dict['o2i'].items():
            if key == '__UNK__' or key == '__MASK__':
                map_dict[x_dict['o2i'][key]] = 0
                continue 
            if key in top_list:
                print(key)
                map_dict[x_dict['o2i'][key]] = 1
            else:
                map_dict[x_dict['o2i'][key]] = 0
        mark = [0] * len(map_dict)
        for key, value in map_dict.items():
            mark[key] = value
    draw_node_emb(mark, save_dir, embedding_model, encode_type, save_name, checkpoint_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To visualize node embedding")
    parser.add_argument("--dataset-dir", type=str, help="np dataset")
    parser.add_argument("--save-dir", type=str, help="The folder to save graph embeddings")
    parser.add_argument("--embedding-model", type=str, default="gcn", help="The model to generater subgraph embedding")
    parser.add_argument("--encode-type", type=str, default="operator", help="The encoder-type: node or operator")
    parser.add_argument("--checkpoint-file", type=str, required=True, help="The selected checkpoint file")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    save_dir = pathlib.Path(args.save_dir)
    checkpoint_file = pathlib.Path(args.checkpoint_file)
    embedding_model = args.embedding_model
    encode_type = args.encode_type
    save_name = f'{embedding_model}_{encode_type}'
    print('now precessing :', save_name)
    node_embedding_process(dataset_dir, save_dir, embedding_model, encode_type, save_name, checkpoint_file)