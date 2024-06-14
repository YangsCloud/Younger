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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import multiprocessing
import matplotlib.pyplot as plt

from typing import Any, Literal, List

from younger.applications.utils.neural_network import get_model_parameters_number, get_device_descriptor, fix_random_procedure, set_deterministic, load_checkpoint, save_checkpoint

from sklearn.manifold import TSNE

from younger.commons.logging import Logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.datasets.node_dataset import NodeDataset
from younger.applications.models import GCN_NP
from younger.applications.datasets import BlockDataset
from younger.applications.tasks.base_task import YoungerTask


def draw_tsne_embedding(datas: List[np.ndarray], save_dir: pathlib.Path, save_name: str, mark: list[int]):
    numpy_array = np.array(datas).squeeze(axis=1)
    print("after-data.shape: ", numpy_array.shape)
    y = np.array(mark)
    print("after Data min:", np.min(numpy_array), "Data max:", np.max(numpy_array))
    print("Contains NaN:", np.isnan(numpy_array).any())
    print("Contains Inf:", np.isinf(numpy_array).any())
    if np.isinf(numpy_array).any():
        print(numpy_array)
        raise ValueError("Input data contains NaN values. Please handle them before using t-SNE.")
    
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(numpy_array)
    tab20_cmap = plt.get_cmap('tab20')
    colors = tab20_cmap.colors
    plt.figure(figsize=(14, 14)) 
    plt.scatter(embedding_2d[y == 0, 0], embedding_2d[y == 0, 1], color=colors[1], label='Other models')
    plt.scatter(embedding_2d[y == 1, 0], embedding_2d[y == 1, 1], color=colors[2], label='resnet-101')
    plt.scatter(embedding_2d[y == 2, 0], embedding_2d[y == 2, 1], color=colors[3], label='resnet-50')
    plt.scatter(embedding_2d[y == 3, 0], embedding_2d[y == 3, 1], color=colors[4], label='resnet-18')
    plt.scatter(embedding_2d[y == 4, 0], embedding_2d[y == 4, 1], color=colors[5], label='roberta')
    plt.scatter(embedding_2d[y == 5, 0], embedding_2d[y == 5, 1], color=colors[6], label='vit-base')
    plt.scatter(embedding_2d[y == 6, 0], embedding_2d[y == 6, 1], color=colors[7], label='vit-large')
    plt.title('t-SNE visualization', fontsize=29)
    plt.tick_params(axis='both', labelsize=28)
    plt.xlabel('Dimension 1', fontsize=28)
    plt.ylabel('Dimension 2', fontsize=28)
    
    plt.legend(fontsize=20, scatterpoints=3)
    plt.savefig(save_dir.joinpath(f'{save_name}_graph_embedding.pdf'),format='pdf')
    plt.savefig(save_dir.joinpath(f'{save_name}_graph_embedding.jpg'),format='jpg')


def graph_process(graph_embedding_dir, save_dir, save_name):
    graph_paths = list(graph_embedding_dir.iterdir())
    datas = list()
    mark = list() 

    for path in tqdm.tqdm(graph_paths):
        graph = torch.load(path)
        if 'resnet' in str(graph['model_names']).split('/')[-1].lower() and '101' in str(graph['model_names']).split('/')[-1].lower():
            mark.append(1)
        elif 'resnet' in str(graph['model_names']).split('/')[-1].lower() and '50' in str(graph['model_names']).split('/')[-1].lower():
            mark.append(2)
        elif 'resnet' in str(graph['model_names']).split('/')[-1].lower() and '18' in str(graph['model_names']).split('/')[-1].lower():
            mark.append(3)
        elif 'roberta' in str(graph['model_names']).split('/')[-1].lower():
            mark.append(4)
        elif 'vit' in str(graph['model_names']).split('/')[-1].lower() and 'base' in str(graph['model_names']).split('/')[-1].lower():
            mark.append(5)
        elif 'vit' in str(graph['model_names']).split('/')[-1].lower() and 'large' in str(graph['model_names']).split('/')[-1].lower():
            mark.append(6)
        else:
            mark.append(0)
        datas.append(graph['graph_embedding'].cpu().numpy())

    draw_tsne_embedding(datas, save_dir, save_name, mark)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To visualize graph embedding")
    parser.add_argument("--graph-embedding-dir", type=str, help="The folder to load graph embedding")
    parser.add_argument("--save-dir", type=str, help="The folder to save graph embeddings")
    parser.add_argument("--embedding-model", type=str, default="gcn", help="The model to generater subgraph embedding")
    parser.add_argument("--encode-type", type=str, default="operator", help="The encoder-type: node or operator")
    args = parser.parse_args()

    graph_embedding_dir = pathlib.Path(args.graph_embedding_dir)
    save_dir = pathlib.Path(args.save_dir)
    embedding_model = args.embedding_model
    encode_type = args.encode_type

    graph_embedding_dir = graph_embedding_dir.joinpath(embedding_model).joinpath(encode_type)
    save_name = f'{embedding_model}_{encode_type}'

    graph_process(graph_embedding_dir, save_dir, save_name)


