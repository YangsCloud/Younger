#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-15 14:07
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import re
import locale

from younger.datasets.utils.detectors import detect_program_langs, detect_natural_langs


def any_pattern_exists(string: str, patterns: list[str]) -> bool:
    exists = False
    for pattern in patterns:
        if re.search(rf'\b{pattern}\b', string):
            exists = True
            break
    return exists


def detect_dataset(string: str) -> str:
    datasets = list()
    has_n_langs = False
    has_p_langs = False
    has_numbers = False

    if any_pattern_exists(string, ['vivos']):
        datasets.append('vivos')

    if any_pattern_exists(string, ['common voice']):
        datasets.append('common-voice')
        has_numbers = True
    
    if any_pattern_exists(string, ['arc', 'ai2 arc', 'arc challenge', 'ai2 reasoning challenge']):
        datasets.append('ai2-reasoning-challenge')

    if any_pattern_exists(string, ['article']):
        datasets.append('article')
        has_numbers = True

    if any_pattern_exists(string, ['aishell']):
        datasets.append('aishell')
        has_n_langs = True

    if any_pattern_exists(string, ['alpacaeval']):
        datasets.append('alpacaeval')
        has_numbers = True

    if any_pattern_exists(string, ['arxiv summarization']):
        datasets.append('arxiv-summarization')

    if any_pattern_exists(string, ['ascend']):
        datasets.append('ascend')

    if any_pattern_exists(string, ['atcosim']):
        datasets.append('atcosim')

    if any_pattern_exists(string, ['audiofolder']):
        datasets.append('audiofolder')

    if any_pattern_exists(string, ['imagefolder', 'image folder']):
        datasets.append('imagefolder')

    if any_pattern_exists(string, ['belebele']):
        datasets.append('belebele')
        has_n_langs = True

    if any_pattern_exists(string, ['bills summarization', 'billsum']):
        datasets.append('billsum')

    if any_pattern_exists(string, ['zeroth korean']):
        datasets.append('zeroth-korean')

    if any_pattern_exists(string, ['callhome spanish speech']):
        datasets.append('callhome-spanish-speech')

    if any_pattern_exists(string, ['chestxrayclassification', 'chest xray']):
        datasets.append('chest-xray')

    if any_pattern_exists(string, ['cifar']):
        datasets.append('cifar')
        has_numbers = True

    if any_pattern_exists(string, ['conllpp']): # Order is Important
        datasets.append('conllpp')
    elif any_pattern_exists(string, ['conll', 'conll2023job']):
        datasets.append('conll')
        has_numbers = True

    if any_pattern_exists(string, ['tweeteval', 'tweet eval']):
        datasets.append('tweeteval')

    if any_pattern_exists(string, ['wmt']):
        datasets.append('wmt')
        has_numbers = True
        has_n_langs = True

    if any_pattern_exists(string, ['wnut']):
        datasets.append('wnut')
        has_numbers = True

    if any_pattern_exists(string, ['xcopa']):
        datasets.append('xcopa')
        has_n_langs = True

    if any_pattern_exists(string, ['xnli']):
        datasets.append('xnli')
        has_n_langs = True

    if any_pattern_exists(string, ['codexglue devign']): # Maybe Different Dataset
        datasets.append('codexglue-devign')
    elif any_pattern_exists(string, ['xglue', 'x glue', 'codexglue']): # Maybe Has Program Language
        datasets.append('codexglue')

    if any_pattern_exists(string, ['multipl humaneval fim']):
        datasets.append('multipl-humaneval-fim')
        has_p_langs = True
    elif any_pattern_exists(string, ['multipl humaneval']):
        datasets.append('multipl-humaneval')
        has_p_langs = True
    elif any_pattern_exists(string, ['humaneval infilling']):
        datasets.append('humaneval infilling')
        has_p_langs = True
    elif any_pattern_exists(string, ['humanevalplus']):
        datasets.append('humanevalplus')
        has_p_langs = True
    elif any_pattern_exists(string, ['humanevalsynthesize']):
        datasets.append('humanevalsynthesize')
        has_p_langs = True
    elif any_pattern_exists(string, ['humanevalexplain']):
        datasets.append('humanevalexplain')
        has_p_langs = True
    elif any_pattern_exists(string, ['humanevalfixdocs']):
        datasets.append('humanevalfixdocs')
        has_p_langs = True
    elif any_pattern_exists(string, ['humanevalfixtests']):
        datasets.append('humanevalfixtests')
        has_p_langs = True
    elif any_pattern_exists(string, ['humanevalfix']):
        datasets.append('humanevalfix')
        has_p_langs = True
    elif any_pattern_exists(string, ['humaneval', 'human eval', 'humanevalpack']):
        datasets.append('humaneval')
        has_p_langs = True

    if any_pattern_exists(string, ['mbpp']):
        datasets.append('mbpp')
        has_p_langs = True

    if any_pattern_exists(string, ['yelp review full']):
        datasets.append('yelp-review-full')

    if any_pattern_exists(string, ['cnn daily', 'cnn dailymail', 'cnndaily']):
        datasets.append('cnn-dailymail')
        has_n_langs = True

    if any_pattern_exists(string, ['restaurant order']):
        datasets.append('restaurant-order')

    if any_pattern_exists(string, ['flores']):
        datasets.append('flores')
        has_numbers = True
        has_n_langs = True

    if any_pattern_exists(string, ['fleurs']): # Need Detect Locale In Future
        datasets.append('fleurs')
        # Simple Replace LOCALE with its Language
        string = re.sub(r'\b([a-z][a-z]) [a-z][a-z]\b', r'\1', string)
        has_n_langs = True

    if any_pattern_exists(string, ['hellaswag']):
        datasets.append('hellaswag')

    if any_pattern_exists(string, ['gsm8k']):
        datasets.append('gsm8k')

    if any_pattern_exists(string, ['indicsuperb', 'indic superb']):
        datasets.append('indicsuperb')

    if any_pattern_exists(string, ['mmlu']):
        datasets.append('mmlu')

    if any_pattern_exists(string, ['mit restaurant', 'mit restaurants']):
        datasets.append('mit-restaurant')

    if any_pattern_exists(string, ['rabbi kook']):
        datasets.append('rabbi-kook')

    if any_pattern_exists(string, ['pawsx', 'paws x']):
        datasets.append('paws-x')
        has_n_langs = True
    elif any_pattern_exists(string, ['paws']):
        datasets.append('paws')

    if any_pattern_exists(string, ['pubmedqa', 'pubmed qa']):
        datasets.append('pubmedqa')

    if any_pattern_exists(string, ['keyword pubmed']):
        datasets.append('keyword-pubmed')

    if any_pattern_exists(string, ['reccon']):
        datasets.append('reccon')

    if any_pattern_exists(string, ['sam']):
        datasets.append('sam')

    if any_pattern_exists(string, ['superglue', 'super glue']):
        datasets.append('superglue')
        has_n_langs = True

    if any_pattern_exists(string, ['truthfulqa', 'truthful qa']):
        datasets.append('truthfulqa')

    if any_pattern_exists(string, ['xstory cloze', 'xstorycloze']):
        datasets.append('xstorycloze')
        has_n_langs = True

    if any_pattern_exists(string, ['winogrande']):
        datasets.append('winogrande')

    if any_pattern_exists(string, ['gdpr recitals']):
        if any_pattern_exists(string, ['en explain']):
            datasets.append('gdpr-recitals-en-explain')
        if any_pattern_exists(string, ['de explain']):
            datasets.append('gdpr-recitals-de-explain')

    if any_pattern_exists(string, ['gdpr articles']):
        if any_pattern_exists(string, ['en explain']):
            datasets.append('gdpr-articles-en-explain')
        if any_pattern_exists(string, ['de explain']):
            datasets.append('gdpr-articles-de-explain')

    if any_pattern_exists(string, ['handbooks']) and any_pattern_exists(string, ['qa']):
        if any_pattern_exists(string, ['en']):
            datasets.append('handbooks-en-qa')
        if any_pattern_exists(string, ['de']):
            datasets.append('handbooks-de-qa')

    if any_pattern_exists(string, ['iso qa']):
        datasets.append('iso-qa')

    if any_pattern_exists(string, ['renovation', 'renovations']):
        datasets.append('renovation')

    if has_n_langs:
        n_langs = list()
        is_show = set()
        for n_lang in detect_natural_langs(string):
            if n_lang in is_show:
                continue
            else:
                n_langs.append(n_lang)
                is_show.add(n_lang)
    else:
        n_langs = list()

    if has_p_langs:
        p_langs = list()
        is_show = set()
        for p_lang in detect_program_langs(string):
            if p_lang in is_show:
                continue
            else:
                p_langs.append(p_lang)
                is_show.add(p_lang)
    else:
        p_langs = list()

    if has_numbers:
        numbers = [number for number in re.findall(r'v?(\d+(?:\.\d+)*)', string) if number.isdigit()]
        numbers = [] if len(numbers) == 1 and numbers[0] == '1' else numbers # Ignore '1' If There's Only A '1'.
        if 'wmt' in datasets or 'wnut' in datasets:
            numbers = [number[-2:] for number in numbers] # 2016 -> 16
    else:
        numbers = list()

    strings = datasets + n_langs + p_langs + numbers
    return ' '.join(strings)