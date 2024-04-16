#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Luzhou Peng (彭路洲) & Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-09 14:03
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import re

from younger.commons.logging import logger


score_0_oo_metrics = set([
    'cider',

    'perplexity',

    'loss',

    'mae',
    'mape',
    'smape',
    'mse',
    'rmse',

    'step',
    'epoch',
    'len',
    'psnr', # [0, 60]
    'is',
    'fid',
    'clip',
    'clipsim',
])


score_0_100_metrics = set([
    'sari',
])


score_0_1_metrics = set([
    'f1',
    'em',
    'map',
    'ap',
    'mc',
    'mrr',
    'win',
    'wip',
    'wil',
    'mer',
    'pass',
    'ndcg',
    'mare',
    'ssim',

    'meteor',
    'sacrebleu',
    'bleu1',
    'bleu2',
    'bleu3',
    'bleu4',
    'rouge-1',
    'rouge-2',
    'rouge-l',
    'rouge-lsum',
    'chrf',
    
    'ser',
    'wer',
    'per',
    'cer',
    'ter',

    'auprc',
    'auroc',

    'mase',
    'mauve',
    'spice',
    'v-measure',

    'accuracy',
    'precision',
    'recall',

    # Special With Certain Single Value
    'brierscore',
    'frugalscore',
    'confusion matrix',
    'code-eval',
])


score_n1_p1_metrics = set([
    'pearson',
    'spearman',
    'matthews',
    'qwk',
])

# approximately 1
score_0_ap1_metrics = set([
    'bleurt',
])

score_special_metrics = {
    'bertscore': [
        'f1',
        'accuracy',
        'precision',
        'recall',
    ],
    'moverscore': [
        'f1',
        'accuracy',
        'precision',
        'recall',
    ],
    'poseval': [
        'f1',
        'accuracy',
        'precision',
        'recall',
    ],
    'seqeval': [
        'f1',
        'accuracy',
        'precision',
        'recall',
    ],
    'super-glue': [
        'f1',
        'em',
        'matthews_correlation',
    ],
    'glue': [
        'f1',
        'accuracy',
        'pearson',
        'spearman',
        'matthews',
    ],
    'wiki-split': [
        'em',
        'sari',
        'sacrebleu',
    ],
    'xtreme-s': [
        'f1',
        'accuracy',
        'wer',
        'cer',
        'bleu4',

    ],
    'squad-v2': [
        'f1',
        'em',
    ],
    'squad': [
        'f1',
        'em',
    ],
}


def detect_basic_metric_adj(string) -> str:
    # Default '' is Micro
    adj = ''
    if 'micro' in string:
        adj = ''
    elif 'macro' in string:
        adj = 'macro'
    elif 'weighted' in string:
        adj = 'weighted'
    elif 'cos sim' in string:
        adj = 'cos sim'
    elif 'manhattan' in string:
        adj = 'manhattan'
    elif 'euclidean' in string:
        adj = 'euclidean'
    elif 'dot' in string:
        adj = 'dot'
    return adj


def detect_basic_metric(string: str) -> str:
    basic_metric = ''
    if 'f1' in string or 'f1score' in string or 'f1 score' in string or 'f measure' in string:
        basic_metric = 'f1'
    elif 'gsm8k' in string:
        basic_metric = 'accuracy'
    elif 'hellaswag' in string or 'winogrande' in string or 'mmlu' in string or 'piqa' in string or 'arc' in string or 'ai 2 reasoning challenge' in string:
        basic_metric = 'accuracy'
    elif 'truthfulqa' in string:
        basic_metric = 'mc'
    elif 'alpacaeval' in string:
        basic_metric = 'win'
    elif 'common voice' in string:
        basic_metric = 'wer'
    elif 'em' in string.split() or 'exact' in string or 'exact match' in string:
        basic_metric = 'em'
    elif 'map' in string.split():
        basic_metric = 'map'
    elif 'ap' in string.split():
        basic_metric = 'ap'
    elif 'mc' in string.split():
        basic_metric = 'mc'
    elif 'win' in string.split():
        basic_metric = 'win'
    elif 'sari' in string.split():
        basic_metric = 'sari'
    elif 'mrr' in string.split():
        basic_metric = 'mrr'
    elif 'pass' in string.split():
        basic_metric = 'pass'
    elif 'ndcg' in string.split():
        basic_metric = 'ndcg'
    elif 'wil' in string.split(): # Word Information Lost
        basic_metric = 'wil'
    elif 'wip' in string.split(): # Word Information Preserved
        basic_metric = 'wip'
    elif 'mer' in string.split(): # Match Error Rate
        basic_metric = 'mer'
    elif 'is' in string.split(): # Inception Score
        basic_metric = 'is'
    elif 'mare' in string.split(): # Mean Absolute Relative Error
        basic_metric = 'mare'
    elif 'fid' in string.split(): # Frechet Inception Distance
        basic_metric = 'fid'
    elif 'psnr' in string.split(): # Peak Signal-to-Noise Ratio
        basic_metric = 'psnr'
    elif 'ssim' in string.split(): # Structural Similarity Index Measure
        basic_metric = 'ssim'
    elif 'clip' in string.split():
        basic_metric = 'clip'
    elif 'clipsim' in string.split(): # CLIP similarity
        basic_metric = 'clipsim'

    # Text
    elif 'meteor' in string:
        basic_metric = 'meteor'
    elif 'sacrebleu' in string:
        basic_metric = 'sacrebleu'
    elif 'bleu' in string:
        match = re.search(r'bleu[\s]?([1,2,3,4]?)', string)
        if match.group(1):
            basic_metric = 'bleu' + match.group(1)
        else:
            basic_metric = 'bleu4'
    elif 'rouge' in string or 'rogue' in string:
        match = re.search(r'(?:rouge|rogue)(?:[\s]?([1,2]|l))?(?:[\s]?(sum))?', string) # May Match Like '1' and 'sum', will be fixed in future.
        if match:
            if match.group(1):
                if match.group(2):
                    basic_metric = 'rouge-' + match.group(1) + match.group(2)
                else:
                    basic_metric = 'rouge-' + match.group(1)
            else:
                basic_metric = 'rouge-1'
    elif 'chrf' in string or 'chr f' in string:
        basic_metric = 'chrf'
    
    elif 'ser' in string.split():
        basic_metric = 'ser'
    elif 'wer' in string.split():
        basic_metric = 'wer'
    elif 'per' in string.split():
        basic_metric = 'per'
    elif 'cer' in string.split():
        basic_metric = 'cer'
    elif 'ter' in string.split():
        basic_metric = 'ter'

    elif 'auprc' in string.split():
        basic_metric = 'auprc'
    elif 'auc' in string.split():
        basic_metric = 'auroc'

    elif 'mase' in string.split():
        basic_metric = 'mase'
    elif 'mauve' in string.split():
        basic_metric = 'mauve'
    elif 'spice' in string.split():
        basic_metric = 'spice'
    elif 'v measure' in string:
        basic_metric = 'v-measure'

    # Correlations
    elif 'matthews' in string:
        basic_metric = 'matthews'
    elif 'pearson' in string:
        basic_metric = 'pearson'
    elif 'spearman' in string:
        basic_metric = 'spearman'
    elif 'qwk' in string:
        basic_metric = 'qwk'

    # Loss
    elif 'cider' in string.split():
        basic_metric = 'cider'
    elif 'perplexity' in string or 'ppl' in string.split():
        basic_metric = 'perplexity'
    elif 'mae' in string.split():
        basic_metric = 'mae-loss'
    elif 'rmse' in string.split():
        basic_metric = 'rmse-loss'
    elif 'mse' in string.split():
        basic_metric = 'mse-loss'
    elif 'smape' in string.split():
        basic_metric = 'smape-loss'
    elif 'mape' in string.split():
        basic_metric = 'mape-loss'
    elif 'loss' in string.split():
        basic_metric = 'loss'

    # Lower Order
    elif 'acc' in string or 'accuracy' in string:
        basic_metric = 'accuracy'
    elif 'precision' in string:
        basic_metric = 'precision'
    elif 'recall' in string:
        basic_metric = 'recall'

    if basic_metric != '':
        adj = detect_basic_metric_adj(string)
        if adj != '':
            basic_metric = basic_metric + ' ' + adj
    return basic_metric


def detect_special_metric(string: str) -> str:
    special_metric = ''
    if 'bertscore' in string or 'bert score' in string:
        special_metric = 'bertscore'
    elif 'moverscore' in string or 'mover score' in string:
        special_metric = 'moverscore'
    elif 'frugalscore' in string or 'frugal score' in string:
        special_metric = 'frugalscore'
    elif 'brierscore' in string or 'brier score' in string:
        special_metric = 'brierscore'
    elif 'poseval' in string:
        special_metric = 'poseval'
    elif 'seqeval' in string:
        special_metric = 'seqeval'
    elif 'squad v2' in string: # Order Is Important
        special_metric = 'squad-v2'
    elif 'squad' in string:
        special_metric = 'squad'
    elif 'super glue' in string: # Order Is Important
        special_metric = 'super-glue'
    elif 'glue' in string:
        special_metric = 'glue'
    elif 'wiki split' in string:
        special_metric = 'wiki-split'
    if 'code eval' in string:
        special_metric = 'code-eval'
    elif 'bleurt' in string:
        special_metric = 'bleurt'
    elif 'confusion matrix' in string: # A Matrix, but We Only Found a Scalar
        special_metric = 'confusion matrix'
    elif 'xtreme s' in string:
        special_metric = 'xtreme-s'

    return special_metric


def detect_metric(string: str) -> str:
    metric = ''

    if 'step' in string:
        metric = 'step'
    elif 'epoch' in string:
        metric = 'epoch'
    elif 'gen len' in string:
        metric = 'len'
    else:
        special_metric = detect_special_metric(string)
        basic_metric = detect_basic_metric(string.replace(special_metric, ''))
        if special_metric in {'bertscore', 'moverscore', 'frugalscore'}:
            basic_metric = 'f1'

        if special_metric == '' and basic_metric != '':
            metric = basic_metric

        if special_metric != '' and basic_metric == '':
            metric = special_metric

        if special_metric != '' and basic_metric != '':
            metric = special_metric + ' ' + basic_metric

    match = re.search(r'@(\d+)', string)
    if match:
        metric = metric + ' ' + match.group(0)

    return metric


def normalize_metric_value(metric: str, metric_value: str) -> float:
    normalized_metric_value = float('NaN')
    origin_metric_value = metric_value.strip()
    base = 1
    if origin_metric_value.endswith('%'):
        origin_metric_value = origin_metric_value[:-1].strip()
        base = base * 100
    elif '+/-' in origin_metric_value:
        origin_metric_value = origin_metric_value.split('+/-')[0].strip()
    elif '\u00b1' in origin_metric_value:
        origin_metric_value = origin_metric_value.split('\u00b1')[0].strip()

    try:
        origin_metric_value = float(origin_metric_value) / base
    except:
        logger.warn(f'Ignored. Invalid Metric Value String, Check It! \'{metric_value}\'')
    
    if isinstance(origin_metric_value, float):
        if len(metric.split()) >= 2:
            main_metric, maybe_main_metric = metric.split()[0], metric.split()[1]
        else:
            main_metric, maybe_main_metric = metric, ''
        
        if main_metric in score_special_metrics:
            if maybe_main_metric in score_special_metrics[main_metric]:
                main_metric, maybe_main_metric = maybe_main_metric, ''
            else:
                main_metric, maybe_main_metric = score_special_metrics[main_metric][0], ''
        
        if main_metric in score_0_100_metrics or maybe_main_metric in score_0_100_metrics:
            normalized_metric_value = origin_metric_value / 100

        elif main_metric in score_0_oo_metrics or maybe_main_metric in score_0_oo_metrics:
            # normalized_metric_value = origin_metric_value / (origin_metric_value + 1)
            # Let Application Choose How to Norm Too Large Number
            normalized_metric_value = origin_metric_value

        elif main_metric in score_n1_p1_metrics or maybe_main_metric in score_n1_p1_metrics:
            normalized_metric_value = (origin_metric_value + 1) / 2

        elif main_metric in score_0_1_metrics or maybe_main_metric in score_0_1_metrics:
            if 1 < origin_metric_value and origin_metric_value <= 100:
                normalized_metric_value = origin_metric_value / 100
            else:
                normalized_metric_value = origin_metric_value
        else:
            normalized_metric_value = origin_metric_value

    return normalized_metric_value