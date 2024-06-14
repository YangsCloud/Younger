from ast import operator
import io
import sys
import time
import json
import numpy
import pathlib
import argparse
import multiprocessing
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


def draw_line_chart_topk_and_vilion(save_dir, operators_full, frequency_full, k, node_count, edge_count):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 16), gridspec_kw={'width_ratios': [34, 66]})

    # ax2.set_xlabel('Operator Types', fontsize=24)
    ax2.set_ylabel('Frequency', fontsize=35)
    ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))

    colorindex = [f'Op{i+1}' for i in range(20)]  
    colors = plt.cm.tab20(np.linspace(0, 1, len(colorindex)))
    colors = np.tile(colors, (len(operators_full[:k]) // len(colors) + 1, 1))[:len(operators_full[:k])]
    ax2.bar(operators_full[:k], frequency_full[:k], alpha=0.6, color=colors)
    ax2.tick_params(axis='both', labelsize=32)
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.set_title('(b)', fontsize=35)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    ax2.set_xticklabels(operators_full[:k], rotation=45, ha='right')
    plt.tight_layout()

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

    violins = ax1.violinplot(
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
    
    ax1.boxplot(
        y_data,
        positions=[0, 1],
        showfliers=False,  
        showcaps=False,  
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops
    )
    
    for x, y, color in zip(x_jittered, y_data, node_colors):
        ax1.scatter(x, y, s=100, color=color, alpha=0.4)

    means = [y.mean() for y in y_data]
    for i, mean in enumerate(means):
        ax1.scatter(i, mean, s=250, color=colors[14], zorder=11)
    
        ax1.plot([i, i + 0.25], [mean, mean], ls="dashdot", color="black", zorder=11)
    
        ax1.text(
            i + 0.25,
            mean,
            r"${\rm{Mean}} = $" + str(round(mean, 2)),
            fontsize=30,
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round",
                pad=0.15,
                alpha=0.5
            ),
            zorder=10  
        )
        
        ax1.tick_params(length=0)
        ax1.set_ylabel("Frequency", size=35)
        # ax1.set_xlabel("", size=30)
        xlabels = [f"{classes}\n" for i, classes in enumerate(classes)]
        ax1.set_xticks([0,1])
        ax1.set_title('(a)', fontsize=35)
        ax1.tick_params(axis='both', labelsize=32)
        ax1.set_xticklabels(xlabels, size=32, ha="center", ma="center")
        ax1.yaxis.labelpad = 25
    
    plt.subplots_adjust(wspace=0.24, left=0.08)

    fig.savefig(save_dir.joinpath(f'villion_bar_top_{k}_frequency.pdf'), format='pdf', dpi=300)
    fig.savefig(save_dir.joinpath(f'villion_bar_top_{k}_frequency.jpg'), format='jpg', dpi=300)


if __name__ == '__main__':
    save_dir = pathlib.Path('/younger/final_experiments/visualization/villion_and_bar')
    dataset_dir = pathlib.Path('/younger/younger/dataset_graph_embedding/node/2024-06-12-00-43-31/initial_full/train/graph')
    worker_number = 20
    node_count = list()
    edge_count = list()
    with open(pathlib.Path('/younger/experiment/Embedding/pic/nodecount_edgecount_new.json'), 'r') as f:
        node_edge_dict = json.load(f)
    print("len(node_edge_dict['edge_count']):", len(node_edge_dict['edge_count']))

    with open(pathlib.Path('/younger/experiment/Embedding/Visualization/satistics_op_sorted_freq.json'), 'r') as f:
        op_freq_sorted_dict = json.load(f)
    
    typical_operators = ['Conv', 'BatchNormalization', 'LayerNormalization', 'Softmax', 'MaxPool',
    'Relu', 'Elu', 'Tanh', 'ReduceSum', 'Pad', 'sigmoid', 'Gemm']
    operators = list()
    operators_full = list()
    frequency = list()
    types = list()
    frequency_full = list()

    for op ,subdict in op_freq_sorted_dict.items():
        if op in typical_operators:
            operators.append(str(op))
            frequency.append(subdict['frequency'])
            types.append(subdict['types'])
        operators_full.append(op)
        frequency_full.append(subdict['frequency'])

    draw_line_chart_topk_and_vilion(save_dir, operators_full, frequency_full, 30, node_edge_dict['node_count'], node_edge_dict['edge_count'])

    