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


def try_clean_at(metric) -> str | None:
    if MetricPattern.AT.search(metric):
        pattern = MetricPattern.AT.pattern
        pattern += try_clean_digit(metric) if try_clean_digit(metric) else ''
        return pattern


def try_clean_split(metric) -> str | None:
    pattern = None
    if MetricPattern.TEST.search(metric):
        pattern = MetricPattern.TEST.pattern
    elif MetricPattern.VALIDATION.search(metric):
        pattern = MetricPattern.VALIDATION.pattern
    elif MetricPattern.TRAIN.search(metric):
        pattern = MetricPattern.TRAIN.pattern
    return pattern


def try_clean_digit(metric) -> str | None:
    digit = MetricPattern.DIGIT.search(metric)
    return digit.group() if digit else ''


def try_clean_f1(metric) -> str | None:
    key = "f1"
    prefix = ''
    if MetricPattern.F1.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        if MetricPattern.MACRO.search(metric):
            prefix += MetricPattern.MACRO.pattern + ' '
        elif MetricPattern.MICRO.search(metric):
            prefix += MetricPattern.MICRO.pattern + ' '
        elif MetricPattern.WEIGHTED.search(metric):
            prefix += MetricPattern.WEIGHTED.pattern + ' '
        else:
            prefix += ""
        key = (prefix + key).strip()
        return key


def try_clean_recall(metric) -> str | None:
    key = "recall"
    prefix = ''
    suffix = ''
    if MetricPattern.RECALL.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        if MetricPattern.MACRO.search(metric):
            prefix += MetricPattern.MACRO.pattern + ' '
        elif MetricPattern.MICRO.search(metric):
            prefix += MetricPattern.MICRO.pattern + ' '
        elif MetricPattern.WEIGHTED.search(metric):
            prefix += MetricPattern.WEIGHTED.pattern + ' '
        else:
            prefix += ""
        suffix += try_clean_at(metric) if try_clean_at(metric) else ''
        key = (prefix + key + suffix).strip()
        return key


def try_clean_precision(metric) -> str | None:
    key = "precision"
    prefix = ''
    suffix = ''
    if MetricPattern.PRECISION.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        if MetricPattern.MACRO.search(metric):
            prefix += MetricPattern.MACRO.pattern + ' '
        elif MetricPattern.MICRO.search(metric):
            prefix += MetricPattern.MICRO.pattern + ' '
        elif MetricPattern.WEIGHTED.search(metric):
            prefix += MetricPattern.WEIGHTED.pattern + ' '
        else:
            prefix += ""
        suffix += try_clean_at(metric) if try_clean_at(metric) else ''
        key = (prefix + key + suffix).strip()
        return key


def try_clean_bleu(metric) -> str | None:
    key = "bleu"
    prefix = ''
    suffix = ''
    if MetricPattern.BLEU.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        if re.compile(r'1', re.IGNORECASE).search(metric):
            suffix += "1"
        if re.compile(r'2', re.IGNORECASE).search(metric):
            suffix += "2"
        if re.compile(r'3', re.IGNORECASE).search(metric):
            suffix += "3"
        if re.compile(r'4', re.IGNORECASE).search(metric):
            suffix += "4"
        else:
            suffix += ""
        key = (prefix + key + suffix).strip()
        return key


def try_clean_rouge(metric) -> str | None:
    key = "rouge"
    prefix = ''
    suffix = ''
    if MetricPattern.ROUGE.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        if re.compile(r'rouge-?1', re.IGNORECASE).search(metric):
            suffix += "1"
        elif re.compile(r'rouge-?lsum', re.IGNORECASE).search(metric):
            suffix += "lsum"
        elif re.compile(r'rouge-?l', re.IGNORECASE).search(metric):
            suffix += "l"
        elif re.compile(r'rouge-?2', re.IGNORECASE).search(metric):
            suffix += "2"
        else:
            suffix += "1"  # perceive rouge as rouge1
        key = (prefix + key + suffix).strip()
        return key


def try_clean_rogue(metric) -> str | None:  # perceive rogue as rouge
    key = "rouge"
    prefix = ''
    suffix = ''
    if MetricPattern.ROGUE.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        if re.compile(r'rogue-?1', re.IGNORECASE).search(metric):
            suffix += "1"
        elif re.compile(r'rogue-?lsum', re.IGNORECASE).search(metric):
            suffix += "lsum"
        elif re.compile(r'rogue-?l', re.IGNORECASE).search(metric):
            suffix += "l"
        elif re.compile(r'rogue-?2', re.IGNORECASE).search(metric):
            suffix += "2"
        else:
            suffix += "1"  # perceive rouge as rouge1
        key = (prefix + key + suffix).strip()
        return key


def try_clean_bertscore(metric) -> str | None:
    key = "bertscore"
    prefix = ''
    if MetricPattern.BERTSCORE.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        key = (prefix + key).strip()
        return key


def try_clean_match(metric) -> str | None:
    key = "match"
    prefix = ''
    if MetricPattern.MATCH.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        key = (prefix + key).strip()
        return key


def try_clean_accuracy(metric) -> str | None:
    key = "accuracy"
    prefix = ''
    if MetricPattern.ACC.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        key = (prefix + key).strip()
        return key


def try_clean_wer(metric) -> str | None:
    key = "wer"
    prefix = ''
    if MetricPattern.WER.search(metric) and "answer" not in metric:
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        key = (prefix + key).strip()
        return key


def try_clean_cer(metric) -> str | None:
    key = "cer"
    prefix = ''
    if MetricPattern.CER.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        key = (prefix + key).strip()
        return key


def try_clean_map(metric) -> str | None:
    key = "map"
    prefix = ''
    suffix = ''
    if MetricPattern.MAP.search(metric):
        prefix += try_clean_split(metric) + ' ' if try_clean_split(metric) else ''
        suffix += try_clean_at(metric) if try_clean_at(metric) else ''
        key = (prefix + key + suffix).strip()
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


def format_metric_string(metric_string: str) -> str:
    metric_string = metric_string.lower()
    return metric_string


def clean_metric(metric_type: str, metric_name: str | None = None, metric_class: str | None = None) -> str | None:
    metric_type = format_metric_string(metric_type)
    metric_name = format_metric_string(metric_name) if metric_name else None
    if metric_name is None:
        detailed_metric = metric_type
    else:
        if len(metric_type) < len(metric_name):
            detailed_metric = metric_name
        else:
            detailed_metric = metric_type

    return parse_metric(detailed_metric)
