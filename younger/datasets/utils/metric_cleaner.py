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
import sys

from younger.datasets.utils.constants import MetricPattern


def try_clean_f1(metric) -> str | None:
    key = "f1"
    pattern = ''
    if MetricPattern.F1.search(metric):
        if MetricPattern.MACRO.search(metric):
            pattern += MetricPattern.MACRO.pattern
        elif MetricPattern.MICRO.search(metric):
            pattern += MetricPattern.MICRO.pattern
        elif MetricPattern.WEIGHTED.search(metric):
            pattern += MetricPattern.WEIGHTED.pattern
        else:
            pattern += ""
        key = (pattern + " " + key).strip()
        return key


def try_clean_recall(metric) -> str | None:
    key = "recall"
    pattern = ''
    if MetricPattern.RECALL.search(metric):
        if MetricPattern.MACRO.search(metric):
            pattern += MetricPattern.MACRO.pattern
        elif MetricPattern.MICRO.search(metric):
            pattern += MetricPattern.MICRO.pattern
        elif MetricPattern.WEIGHTED.search(metric):
            pattern += MetricPattern.WEIGHTED.pattern
        else:
            pattern += ""
        key = (pattern + " " + key).strip()
        return key


def try_clean_precision(metric) -> str | None:
    key = "precision"
    pattern = ''
    if MetricPattern.PRECISION.search(metric):
        if MetricPattern.MACRO.search(metric):
            pattern += MetricPattern.MACRO.pattern
        elif MetricPattern.MICRO.search(metric):
            pattern += MetricPattern.MICRO.pattern
        elif MetricPattern.WEIGHTED.search(metric):
            pattern += MetricPattern.WEIGHTED.pattern
        else:
            pattern += ""
        key = (pattern + " " + key).strip()
        return key


def try_clean_bleu(metric) -> str | None:
    key = "bleu"
    pattern = ''
    if MetricPattern.BLEU.search(metric):
        if re.compile(r'1', re.IGNORECASE).search(metric):
            pattern += "1"
        if re.compile(r'2', re.IGNORECASE).search(metric):
            pattern += "2"
        if re.compile(r'3', re.IGNORECASE).search(metric):
            pattern += "3"
        if re.compile(r'4', re.IGNORECASE).search(metric):
            pattern += "4"
        else:
            pattern += ""
        key = (key + pattern).strip()
        return key


def try_clean_rouge(metric) -> str | None:
    key = "rouge"
    pattern = ''
    if MetricPattern.ROUGE.search(metric):
        if re.compile(r'rouge-?1', re.IGNORECASE).search(metric):
            pattern += "1"
        elif re.compile(r'rouge-?lsum', re.IGNORECASE).search(metric):
            pattern += "lsum"
        elif re.compile(r'rouge-?l', re.IGNORECASE).search(metric):
            pattern += "l"
        elif re.compile(r'rouge-?2', re.IGNORECASE).search(metric):
            pattern += "2"
        else:
            pattern += "1"  # perceive rouge as rouge1
        key = (key + pattern).strip()
        return key


def try_clean_rogue(metric) -> str | None:  # perceive rogue as rouge
    key = "rouge"
    pattern = ''
    if MetricPattern.ROGUE.search(metric):
        if re.compile(r'rogue-?1', re.IGNORECASE).search(metric):
            pattern += "1"
        elif re.compile(r'rogue-?lsum', re.IGNORECASE).search(metric):
            pattern += "lsum"
        elif re.compile(r'rogue-?l', re.IGNORECASE).search(metric):
            pattern += "l"
        elif re.compile(r'rogue-?2', re.IGNORECASE).search(metric):
            pattern += "2"
        else:
            pattern += "1"  # perceive rouge as rouge1
        key = (key + pattern).strip()
        return key


def try_clean_bertscore(metric) -> str | None:
    key = "bertscore"
    pattern = ''
    if MetricPattern.BERTSCORE.search(metric):
        key = (pattern + " " + key).strip()
        return key


def try_clean_match(metric) -> str | None:
    key = "match"
    pattern = ''
    if MetricPattern.MATCH.search(metric):
        key = (pattern + " " + key).strip()
        return key


def try_clean_accuracy(metric) -> str | None:
    key = "accuracy"
    pattern = ''
    if MetricPattern.ACC.search(metric):
        key = (pattern + " " + key).strip()
        return key


def try_clean_wer(metric) -> str | None:
    key = "wer"
    pattern = ''
    if MetricPattern.WER.search(metric) and "answer" not in metric:
        key = (pattern + " " + key).strip()
        return key


def try_clean_cer(metric) -> str | None:
    key = "cer"
    pattern = ''
    if MetricPattern.CER.search(metric):
        key = (pattern + " " + key).strip()
        return key


def try_clean_map(metric) -> str | None:
    key = "map"
    pattern = ''
    if MetricPattern.MAP.search(metric):
        key = (pattern + " " + key).strip()
        return key


def parse_metric(metric: str) -> str | None:
    clean_pipeline = [
        'f1',
        'recall',
        'precision',
        'bleu',
        'rouge',
        'rogue',
        'bertscore',
        'match',
        'accuracy',
        'wer',
        'cer',
        'map'
    ]

    parsed_result = metric
    for clean_work_name in clean_pipeline:
        clean_work_method = getattr(sys.modules[__name__], f'try_clean_{clean_work_name}')
        cleaned_result = clean_work_method(metric)
        if cleaned_result is not None:
            parsed_result = cleaned_result
            break
    return parsed_result


def clean_metric(metric_type: str, metric_name: str | None = None, metric_class: str | None = None) -> str | None:
    metric_type = metric_type.lower()
    metric_name = metric_name.lower()
    if metric_name is None:
        detailed_metric = metric_type
    else:
        if len(metric_type) < len(metric_name):
            detailed_metric = metric_name
        else:
            detailed_metric = metric_type

    detailed_metric = metric_name
    return parse_metric(detailed_metric)