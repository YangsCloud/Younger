from platform import node
import re
import torch
import pathlib
import ast
import json
import numpy as np
import tqdm
import torch
import torch.utils.data
import multiprocessing
import pandas as pd
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Literal, List
from collections import OrderedDict
from torch_geometric.data import Batch, Data

from younger.applications.utils.neural_network import get_model_parameters_number, get_device_descriptor, fix_random_procedure, set_deterministic, load_checkpoint, save_checkpoint
from younger.commons.io import load_pickle
from sklearn.manifold import TSNE

from younger.commons.logging import Logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.datasets.node_dataset import NodeDataset
from younger.applications.models import GLASS, GCN_NP
from younger.applications.datasets import BlockDataset
from younger.applications.tasks.base_task import YoungerTask



def draw_villion(save_dir, node_count, edge_count):
    data = {
    'Value': node_count + edge_count,
    'Group': ['Node'] * len(node_count) + ['Edge'] * len(edge_count)
    }
    df = pd.DataFrame(data)
    classes = sorted(df["Group"].unique())

    y_data = [df[df["Group"] == c]["Value"].values for c in classes]
    jitter = 0.04

    x_data = [np.array([i] * len(d)) for i, d in enumerate(y_data)]
    x_jittered = [x + st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]
    
    tab20_cmap = plt.get_cmap('tab20')
    colors = tab20_cmap.colors

    fig, ax = plt.subplots(figsize=(14, 10))

    violins = ax.violinplot(
        y_data,
        positions=[0, 1],
        widths=0.45,
        bw_method="silverman",
        showmeans=False,
        showmedians=False,
    )
    node_colors = [colors[2],colors[18]]

    for pc in violins["bodies"]:
        pc.set_zorder(10)
        pc.set_facecolor("none")
        pc.set_edgecolor(colors[10])
        pc.set_linewidth(1.4)
        pc.set_alpha(1)
    
    medianprops = dict(
        linewidth=4,
        color=colors[1],
        solid_capstyle="butt"
    )
    boxprops = dict(
        linewidth=3,
        color=colors[3]
    )
    
    ax.boxplot(
        y_data,
        positions=[0, 1],
        showfliers=False,  
        showcaps=False,  
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops
    )
    
    for x, y, color in zip(x_jittered, y_data, node_colors):
        ax.scatter(x, y, s=100, color=color, alpha=0.4)

    means = [y.mean() for y in y_data]
    for i, mean in enumerate(means):
        ax.scatter(i, mean, s=250, color=colors[14], zorder=3)
    
        ax.plot([i, i + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    
        ax.text(
            i + 0.25,
            mean,
            r"${\rm{Mean}} = $" + str(round(mean, 2)),
            fontsize=23,
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round",
                pad=0.15
            ),
            zorder=10  
        )
        
        ax.tick_params(length=0)
        ax.set_ylabel("Frequency", size=30)
        xlabels = [f"{classes}\n" for i, classes in enumerate(classes)]
        ax.set_xticks([0,1])
        ax.tick_params(axis='both', labelsize=22)
        ax.set_xticklabels(xlabels, size=28, ha="center", ma="center")


    plt.savefig(save_dir.joinpath(f'villion_node_edge_new.pdf'), format='pdf')
    plt.savefig(save_dir.joinpath(f'villion_node_edge_new.jpg'), format='jpg')


if __name__ == '__main__':

    save_dir = pathlib.Path('/younger/experiment/Embedding/pic')
    dataset_dir = pathlib.Path('/younger/younger/dataset_graph_embedding/node/2024-06-12-00-43-31/initial_full/train/graph')
    worker_number = 20

    node_count = list()
    edge_count = list()
    instances = list(dataset_dir.iterdir())
    
    # if worker_number is not None:
    #     with multiprocessing.Pool(worker_number) as pool:
    #         with tqdm.tqdm(total=len(instances), desc='Filtering') as progress_bar:
    #             for index, instance in enumerate(pool.imap_unordered(load_pickle, instances), start=1):
    #                 node_count.append(len(instance.nodes))
    #                 edge_count.append(len(instance.edges))
    #                 progress_bar.update(1)
    # else:
    #     for instance_path in tqdm.tqdm(instances):
    #         instance = load_pickle(instance_path)
    #         node_count.append(len(instance.nodes))
    #         edge_count.append(len(instance.edges))

    # data = dict(
    #     node_count = node_count,
    #     edge_count = edge_count,
    # )
    # with open(save_dir.joinpath('nodecount_edgecount_new.json'), 'w') as f:
    #     json.dump(data, f)

    with open(save_dir.joinpath('nodecount_edgecount_new.json'), 'r') as f:
        node_edge_dict = json.load(f)
    print(len(node_edge_dict['edge_count']))
    draw_villion(save_dir, node_edge_dict['node_count'], node_edge_dict['edge_count'])