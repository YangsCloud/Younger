from ast import operator
import io
import sys
import time
import json
import numpy
import pathlib
import argparse
import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def draw_typical_bars(save_dir: pathlib.Path, operators: list[str], frequency: list[int], types: list[int]):
    
    frequency = [element/100 for element in frequency]

    bar_width = 0.35
    index = np.arange(len(operators))

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('operators')
    ax1.set_ylabel('frequency')
    bars1 = ax1.bar(index, frequency, bar_width, label='frequency / 100', color='blue')
    ax1.set_xticks(index)
    ax1.set_xticklabels(operators)

    ax2 = ax1.twinx() 
    ax2.set_ylabel('types')
    bars2 = ax2.bar(index + bar_width, types, bar_width, label='counts', color='green')

    plt.title('title')

    bars = bars1 + bars2
    labels = [bar.get_label() for bar in bars]
    ax1.legend(bars, labels, loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig.savefig(save_dir.joinpath('Some_typical_Frequency_types.pdf'), format='pdf', dpi=300)


def draw_typical_frequency(save_dir, operators, frequency):

    index = np.arange(len(operators))
    bar_width = 0.75
    colors = plt.cm.tab20(np.linspace(0, 1, len(operators)))
    fig, ax1 = plt.subplots(figsize=(10, 8))

    ax1.set_xlabel('Operator Types', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.bar(index, frequency, bar_width, label='Frequency', alpha=0.6, color=colors)

    ax1.set_xticks(index)
    ax1.set_xticklabels(operators)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(save_dir.joinpath('Some_typical_Frequency.pdf'), format='pdf', dpi=300)

def draw_line_chart_topk(save_dir, operators_full, frequency_full, k):
    fig, ax1 = plt.subplots(figsize=(17, 12))

    ax1.set_xlabel('Operator Types', fontsize=24)
    ax1.set_ylabel('Frequency', fontsize=24)
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))

    colorindex = [f'Op{i+1}' for i in range(20)]  
    colors = plt.cm.tab20(np.linspace(0, 1, len(colorindex)))
    colors = np.tile(colors, (len(operators_full[:k]) // len(colors) + 1, 1))[:len(operators_full[:k])]
    ax1.bar(operators_full[:k], frequency_full[:k], alpha=0.6, color=colors)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.ticklabel_format(style='plain', axis='y')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(save_dir.joinpath(f'top_{k}_frequency.pdf'), format='pdf', dpi=300)
    fig.savefig(save_dir.joinpath(f'top_{k}_frequency.jpg'), format='jpg', dpi=300)


if __name__ == '__main__':
    
    save_dir = pathlib.Path('/younger/experiment/Embedding/Visualization')

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

    draw_typical_frequency(save_dir, operators, frequency)
    draw_typical_bars(save_dir, operators, frequency, types)
    draw_line_chart_topk(save_dir, operators_full, frequency_full, k=20)
    draw_line_chart_topk(save_dir, operators_full, frequency_full, k=30)
    draw_line_chart_topk(save_dir, operators_full, frequency_full, k=40)

    