from importlib.resources import path
import re
import torch
import pathlib
import ast
import json
import argparse
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
from younger.applications.models import GLASS, GCN_NP
from younger.applications.datasets import BlockDataset
from younger.applications.tasks.base_task import YoungerTask


def draw_tsne_embedding(datas: List[np.ndarray], save_dir, save_name: str):
    numpy_array = np.array(datas).squeeze(axis=1)
    print("after-data.shape: ", numpy_array.shape)
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
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], color=colors[1])
    plt.title('t-SNE visualization', fontsize=29)
    plt.tick_params(axis='both', labelsize=28)
    plt.xlabel('Dimension 1', fontsize=28)
    plt.ylabel('Dimension 2', fontsize=28)
    
    plt.savefig(save_dir.joinpath(f'{save_name}_subgraph_embedding.pdf'),format='pdf')
    plt.savefig(save_dir.joinpath(f'{save_name}_subgraph_embedding.jpg'),format='jpg')


def subgraph_process(subgraph_embedding_dir: pathlib.Path, save_dir: pathlib.Path, save_name: str):
    datas = list()
    paths = list(subgraph_embedding_dir.iterdir())
    for path in tqdm.tqdm(paths):
        if '.json' in str(path):
            continue
        subgraph = torch.load(path)
        datas.append(subgraph['embedding'].cpu().numpy())
            
    draw_tsne_embedding(datas, save_dir, save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To visualize subgraph embedding")
    parser.add_argument("--subgraph-embedding-dir", type=str, help="The folder to load graph embedding")
    parser.add_argument("--save-dir", type=str, help="The folder to save graph embeddings")
    parser.add_argument("--embedding-model", type=str, default="gcn", help="The model to generater subgraph embedding")
    parser.add_argument("--encode-type", type=str, default="operator", help="The encoder-type: node or operator")
    args = parser.parse_args()

    subgraph_embedding_dir = pathlib.Path(args.subgraph_embedding_dir)
    save_dir = pathlib.Path(args.save_dir)
    embedding_model = args.embedding_model
    encode_type = args.encode_type

    subgraph_embedding_dir = subgraph_embedding_dir.joinpath(embedding_model).joinpath(encode_type)
    save_name = f'{embedding_model}_{encode_type}'
    subgraph_process(subgraph_embedding_dir, save_dir, save_name)