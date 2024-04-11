#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-12 00:22
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re

from younger.commons.constants import Constant


class METRIC_PATTERN(Constant):
    def initialize(self) -> None:
        self.ACC = re.compile(r'acc', re.IGNORECASE)
        self.F1 = re.compile(r'f1', re.IGNORECASE)
        self.RECALL = re.compile(r'recall', re.IGNORECASE)
        self.PRECISION = re.compile(r'precision', re.IGNORECASE)
        self.BLEU = re.compile(r'bleu', re.IGNORECASE)
        self.ROUGE = re.compile(r'rouge-?', re.IGNORECASE)
        self.ROGUE = re.compile(r'rogue-?', re.IGNORECASE)
        self.BERTSCORE = re.compile(r'bertscore', re.IGNORECASE)
        self.WER = re.compile(r'wer', re.IGNORECASE)
        self.CER = re.compile(r'cer', re.IGNORECASE)
        self.MAP = re.compile(r'map', re.IGNORECASE)
        self.MATCH = re.compile(r'match', re.IGNORECASE)

        self.MACRO = re.compile(r'macro', re.IGNORECASE)
        self.MICRO = re.compile(r'micro', re.IGNORECASE)
        self.WEIGHTED = re.compile(r'weighted', re.IGNORECASE)
        self.AT = re.compile(r'_at_', re.IGNORECASE)
        self.DIGIT = re.compile(r'\d+', re.IGNORECASE)

        self.TEST = re.compile(r'test', re.IGNORECASE)
        self.VALIDATION = re.compile(r'validation', re.IGNORECASE)
        self.TRAIN = re.compile(r'train', re.IGNORECASE)
        self.EVAL = re.compile(r'eval', re.IGNORECASE)

MetricPattern = METRIC_PATTERN()
MetricPattern.initialize()
MetricPattern.freeze()