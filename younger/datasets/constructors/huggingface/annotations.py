#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-12 00:26
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re

from typing import Any, Literal

from huggingface_hub import ModelCardData

from younger.commons.logging import logger

from younger.datasets.utils.detectors import detect_task, detect_dataset, detect_split, detect_metric, normalize_metric_value


def get_detailed_string(strings: list[str]) -> str:
    return max(strings, key=len)


def split_camel_case_string(camel_case_string: str) -> list[str]:
    words = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', camel_case_string)
    return [word.group(0) for word in words]


def clean_string(string: str, split_camel_case: bool = False) -> str:
    # 0. Remove Sub-string Before The First '/'
    string = re.sub(r'^.*?/', '', string, 1)

    # 1. Split All Camel Case Words:
    #   Metrics like CIDEr, ROUGEl, HellaSwag, BERTScore, ... Can Be Splited
    #   So we set split_camel_case to be False
    if split_camel_case:
        words = list()
        for word in string.split():
            words.extend(split_camel_case_string(word))
            string = ' '.join(words)

    # 3. Remove All Parentheses:
    note_pattern = r'\(([^()]*?)\)'
    note_strings = [note_string.lower() for note_string in re.findall(note_pattern, string)]
    string = re.sub(note_pattern, '', string)
    strings = [string] + note_strings
    tidy_string = ' '.join(strings)

    # 3. Convert '_at_' -> '@':
    tidy_string = tidy_string.replace('_at_', '@')

    # 4. Clean All '_'  '-'  ':'  '.'  ',' Chars:
    tidy_string = tidy_string.replace('_', ' ')
    tidy_string = tidy_string.replace('-', ' ')
    tidy_string = tidy_string.replace(':', ' ')
    tidy_string = tidy_string.replace('.', ' ')
    tidy_string = tidy_string.replace(',', ' ')

    # 5. Seperate Version String & Clean Version Indicator 'v': (Like: 'v1.1.1' -> '1.1.1', 'article500v0' -> 'article500 0')
    tidy_string = re.sub(r'([^a-zA-Z]+)v(\d+)', r'\1 \2', tidy_string)

    # 6. Seperate Number Right Next To A Word [Except 'QAv1', 'LMv1', 'f1'...]: (Like: 'article500' -> 'article 500', 'p2p' -> 'p2p', '4K' -> '4K')
    # tidy_string = re.sub(r'(\D+)(\d+)(?=\s|$)', r'\1 \2', tidy_string)
    tidy_string = re.sub(r'(\D+[A-Z])?([v]?@?[fF]?\d+)|([a-zA-Z]+)([v]?@?[fF]?\d+)', r'\1\3 \2\4', tidy_string)

    # 7. Clean Additional Spaces:
    tidy_string = ' '.join(tidy_string.split())

    # 8. Lower All Chars:
    tidy_string = tidy_string.lower()

    return tidy_string


def parse_task(task_name: str) -> str:
    task_name = task_name.replace('text2text', 'text-to-text')
    string = clean_string(task_name)

    # Correcting Spelling Errors
    string = string.replace('label', ' class ')
    string = string.replace('multiple', ' multi ')
    string = string.replace('qa', ' question answering ')
    string = string.replace('lenguage', 'language')
    string = string.replace('modelling', 'modeling')
    string = string.replace('classfication', 'classification')

    detected_task_name = detect_task(string)
    if detected_task_name == '':
        detected_task_name = string
    else:
        detected_task_name = detected_task_name

    detected_task_name = ' '.join(detected_task_name.split())
    return detected_task_name


def parse_dataset(dataset_names: list[str]) -> tuple[str, Literal['train', 'valid', 'test']]:
    detected_dataset_names = list()
    strings = list()
    for dataset_name in dataset_names:
        string = clean_string(dataset_name)

        # Correcting Spelling Errors
        string = string.replace('common voices', 'common voice')
        string = string.replace('commonvoice', 'common voice')
        string = string.replace('invoices', 'invoice')
        string = string.replace('humanneval', 'humaneval')

        # Extract Shots: '1 shot'
        shot_pattern = r'\b((?:\d+~)?(?:\d+) shot)\b'
        shots = ' '.join(re.findall(shot_pattern, string))

        string = re.sub(shot_pattern, '', string)

        detected_dataset_names.append(detect_dataset(string))
        strings.append(string)

    detected_dataset_name = max(detected_dataset_names, key=len)
    string = max(strings, key=len)

    if detected_dataset_name == '':
        strings = [string, shots]
        detected_dataset_name = ' '.join(strings)
    else:
        strings = [detected_dataset_name, shots]
        detected_dataset_name = ' '.join(strings)

    detected_dataset_name = ' '.join(detected_dataset_name.split())
    return detected_dataset_name


def parse_metric(metric_name: str):
    if metric_name.lower() in {'n/a', 'n.a.'}:
        return ''
    string = clean_string(metric_name)
    string = string.replace(' language model', ' lm')
    string = string.replace(' no ', ' -')
    string = string.replace(' w/o ', ' -')
    string = string.replace(' without ', ' -')
    string = string.replace(' with ', ' +')
    string = string.replace(' using ', ' +')
    # Skip +/- words
    # switch_words = list()
    # for switch_word in mn_string.split():
    #     if switch_word.startswith('+'):
    #         switch_words.append(switch_word)
    # print(switch_words)
    detected_metric_name = detect_metric(string)

    if detected_metric_name == '':
        detected_metric_name = string
    else:
        detected_metric_name = detected_metric_name

    detected_metric_name = ' '.join(detected_metric_name.split())
    return detected_metric_name


def get_heuristic_annotations(model_id: str, model_card_data: ModelCardData) -> list[dict[str, dict[str, str]]] | None:
    if model_card_data.eval_results:
        annotations = list()
        for eval_result in model_card_data.eval_results:
            hf_task_type = eval_result.task_type
            hf_task_name = eval_result.task_name if eval_result.task_name else ''

            # Skip dataset_conig & dataset_args
            # The infomation provided by config & args mostly appear in dataset_type & dataset_name
            # Or it is not a good ModelCard & Model Repo, thus model may useless for our project.
            hf_dataset_type = eval_result.dataset_type
            hf_dataset_name = eval_result.dataset_name
            hf_dataset_split  = eval_result.dataset_split if eval_result.dataset_split else ''

            # Skip metric_conig & metric_args
            # The infomation provided by config & args mostly appear in hf_metric_type & hf_metric_name
            # Or it is not a good ModelCard & Model Repo, thus model may useless for our project.
            hf_metric_type = eval_result.metric_type
            hf_metric_value = eval_result.metric_value
            hf_metric_name = eval_result.metric_name if eval_result.metric_name else ''

            # A. For a Task Name: In most cases there should not be a 'Note Strings' i.e. '(note string)s'
            #    If there is, It MIGHT Be a Dataset Name!
            #    So, if any word in dataset_name or dataset_type appear in task_name -> task_name indicates a dataset_name & task_type definitely is a task_type.
            # B. Besides, dataset_type <> task_type, if they are equal, there MUST BE an error occur when authors write the ModelCard.
            #    So, task_name is dataset_name, like example 3
            # See Examples:
            # 1. https://huggingface.co/relbert/relbert-albert-base-nce-semeval2012/edit/main/README.md
            #    v 'analogy', 'questions', 'sat full' in task_name
            #    task:
            #      name: Analogy Questions (SAT full)
            #      type: multiple-choice-qa
            #    dataset:
            #      name: SAT full
            #      type: analogy-questions
            # 2. https://huggingface.co/gsarti/mt5-small-repubblica-to-ilgiornale/edit/main/README.md
            #    v 'eurlex' in task_name
            #    task:
            #      type: text-classification
            #      name: Danish EURLEX (Level 2)
            #    dataset:
            #      name: multi_eurlex
            #      type: multi_eurlex
            # 3. https://huggingface.co/TheBloke/juanako-7B-UNA-GPTQ/edit/main/README.md
            #    v 'truthfull', 'qa' in task_name
            #    task:
            #      name: TruthfulQA (MC2)
            #      type: text-generation
            #    dataset:
            #      name: truthful_qa
            #      type: text-generation
            # 4. https://huggingface.co/gsarti/mt5-small-repubblica-to-ilgiornale/blob/main/README.md
            #    v This is a phenomenon that doesn't match the situation
            #    task: 
            #      type: headline-style-transfer-repubblica-to-ilgiornale
            #      name: "Headline style transfer (Repubblica to Il Giornale)"
            #    dataset:
            #      type: gsarti/change_it
            #      name: "CHANGE-IT"
            if hf_task_type == hf_dataset_type:
                logger.warn(f'Annotations Maybe Wrong. Try To Fix: Model ID {model_id}')
                hf_task_name, hf_dataset_name = hf_task_type, hf_task_name
                hf_task_type, hf_dataset_type = '', ''
            elif hf_dataset_name in hf_task_type:
                hf_task_type, hf_dataset_type = hf_dataset_type, hf_task_type
            else:
                if re.search(r'\(([^()]*?)\)', hf_task_name):
                    # dn_string = clean_string(hf_dataset_name)
                    # dt_string = clean_string(hf_dataset_type)
                    words = [word for word in hf_dataset_name.split()] + [word for word in hf_dataset_type.split()]
                    for word in words:
                        if word in hf_task_name:
                            logger.warn(f'Annotations Maybe Wrong. Try To Fix: Model ID {model_id}')
                            hf_task_type, hf_dataset_name = hf_dataset_type, hf_task_name
                            hf_task_name, hf_dataset_type = '', ''
                            break

            # detailed_task_name = get_detailed_string([hf_task_name, hf_task_type])
            # task_name = parse_task(detailed_task_name)
            task_name = get_detailed_string([parse_task(hf_task_name), parse_task(hf_task_type)])

            detailed_dataset_name = get_detailed_string([hf_dataset_type, hf_dataset_name])
            # dataset_name = parse_dataset(detailed_dataset_name)
            dataset_name = parse_dataset([hf_dataset_name, hf_dataset_type])

            if dataset_name in task_name:
                dataset_name, task_name = task_name, dataset_name

            split = detect_split(hf_dataset_split)
            if split == '':
                split = detect_split(clean_string(detailed_dataset_name))

            detailed_metric_name = get_detailed_string([hf_metric_type, hf_metric_name])
            if isinstance(hf_metric_value, list):
                logger.warn(f'Skip. Useless Metric Value. Model ID {model_id} {eval_result.metric_value}')
                metric_info = None
                candidate_hf_metrics = list()
            elif isinstance(hf_metric_value, dict):
                candidate_hf_metrics = list()
                for k, v in hf_metric_value.items():
                    if k.isdigit():
                        continue
                    else:
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                candidate_hf_metrics.append((detailed_metric_name+' '+k+' '+kk, str(vv)))
                        else:
                            candidate_hf_metrics.append((detailed_metric_name+' '+k, str(v)))
            else:
                candidate_hf_metrics = [(detailed_metric_name, str(hf_metric_value))]

            for mname, mvalue in candidate_hf_metrics:
                if split == '':
                    split = detect_split(clean_string(mname))

                parsed_mname = parse_metric(mname)
                parsed_mname = mname if parsed_mname == '' else parsed_mname
                norm_mvalue = normalize_metric_value(parsed_mname, mvalue)
                metric_info = (parsed_mname, norm_mvalue)

                if split == '':
                    split = 'test'
                dataset_info = (dataset_name, split)

                annotations.append(
                    dict(
                        task=task_name,
                        dataset=dataset_info,
                        metric=metric_info,
                    )
                )
    else:
        annotations = None
    
    return annotations


def get_manually_annotations():
    pass