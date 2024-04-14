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

from younger.datasets.utils.detectors import detect_program_langs, detect_natural_langs, detect_metric, detect_split, normalize_metric_value


def get_detailed_string(strings: list[str]) -> str:
    return max(strings, key=len)


def split_camel_case_string(camel_case_string: str) -> list[str]:
    words = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', camel_case_string)
    return [word.group(0) for word in words]


def initial_parse_string(string: str) -> tuple[str, list[str], list[str]]:
    # 1. Split All Camel Case Words:
    words = list()
    for word in string.split():
        words.extend(split_camel_case_string(word))
    string = ' '.join(words)

    # 2. Get All Note Strings Wrapped in Parentheses:
    note_pattern = r'\(([^()]*?)\)'
    note_strings = re.findall(note_pattern, string)

    # 3. Remove Note Strings
    tidy_string = re.sub(note_pattern, '', string)

    # 4. Clean All '_' Chars:
    tidy_string = tidy_string.replace('_', ' ')

    # 5. Clean Additional Spaces:
    tidy_string = ' '.join(tidy_string.split())

    # 6. Lower All Chars:
    tidy_string = tidy_string.lower()

    # 7. Split Main String and Other Strings:
    tidy_strings = tidy_string.split(' - ')
    main_string = tidy_strings[0]
    if len(tidy_strings) > 1:
        other_strings = tidy_strings[1:]
    else:
        other_strings = list()

    return main_string, note_strings, other_strings


def parse_task(task_name: str) -> str:
    main_string, note_strings, other_strings = initial_parse_string(task_name)
    strings = main_string.split() + note_strings + other_strings
    tn_string = ' '.join(strings)

    tn_string = tn_string.replace('-', ' ')
    tn_string = tn_string.replace('label', ' class ')
    tn_string = tn_string.replace('multiple', ' multi ')
    tn_string = tn_string.replace('qa', ' question answering ')

    return tn_string


def parse_dataset(dataset_name: str) -> tuple[str, Literal['train', 'valid', 'test']]:
    main_string, note_strings, other_strings = initial_parse_string(dataset_name)
    strings = main_string.split() + note_strings + other_strings
    dn_string = ' '.join(strings)

    if 'humaneval' in dn_string:
        # No More Process
        plangs = detect_program_langs(dn_string)

    elif 'voice' in dn_string:
        # Not Search All Version String, Just Replace All '.' with ' '
        # re.findall(r'(v?\d+(?:\.\d+)*)')
        dn_string = dn_string.replace('-', ' ')
        dn_string = dn_string.replace('.', ' ')
        dn_string = dn_string.replace(',', ' ')

        dn_string = dn_string.replace('common voices', 'common voice')
        dn_string = dn_string.replace('commonvoice', 'common voice')
        dn_string = dn_string.replace('invoices', 'invoice')

        # Only Preserve Natural Languages & Versions
        # Ignore '+' Case:
        #   Example: 'vivos + commonvoice'
        if (
            (dn_string.startswith('common voice')) or
            (len(dn_string.split('/')) == 2 and dn_string.split('/')[1].startswith('common voice')) or
            ('common voice' in dn_string and dn_string.startswith('mozilla')
        )):
            if len(dn_string.split('/')) >= 2:
                dn_string = dn_string.split('/')[1]
            nlangs = detect_natural_langs(dn_string)
            nums = [num for num in re.findall(r'v?(\d+(?:\.\d+)*)', dn_string) if int(num)]
            dn_strings = ['common voice'] + nums + nlangs
            dn_string = ' '.join(dn_strings)

    dn_string = dn_string.replace('/', ' ')

    return dn_string


def parse_metric(metric_name: str):
    main_string, note_strings, other_strings = initial_parse_string(metric_name)
    strings = main_string.split() + note_strings + other_strings
    mn_string = ' '.join(strings)
    mn_string = mn_string.replace('language model', 'lm')
    mn_string = mn_string.replace('no ', '-')
    mn_string = mn_string.replace('w/o ', '-')
    mn_string = mn_string.replace('without ', '-')
    mn_string = mn_string.replace('with ', '+')
    mn_string = mn_string.replace('using ', '+')
    mn_string = mn_string.replace(' at ', ' @')
    # Skip +/- words
    # switch_words = list()
    # for switch_word in mn_string.split():
    #     if switch_word.startswith('+'):
    #         switch_words.append(switch_word)
    # print(switch_words)
    mn_string = detect_metric(mn_string)

    return mn_string


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

            # A. For a Task Name: In most cases there should not be a 'Note Strings'
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
            else:
                _, tn_notes, _ = initial_parse_string(hf_task_name)
                if len(tn_notes):
                    dn_main, dn_notes, dn_others = initial_parse_string(hf_dataset_name)
                    dt_main, dt_notes, dt_others = initial_parse_string(hf_dataset_type)
                    words = (
                        [word for word in dn_main.split()] +
                        [word for dn_note in dn_notes for word in dn_note.split()] +
                        [word for dn_other in dn_others for word in dn_other.split()] +
                        [word for word in dt_main.split()] +
                        [word for dt_note in dt_notes for word in dt_note.split()] +
                        [word for dt_other in dt_others for word in dt_other.split()]
                    )
                    for word in words:
                        if word in hf_task_name:
                            logger.warn(f'Annotations Maybe Wrong. Try To Fix: Model ID {model_id}')
                            hf_task_name, hf_dataset_name = hf_task_type, hf_task_name
                            hf_task_type, hf_dataset_type = '', ''
                            break

            task_name = parse_task(get_detailed_string([hf_task_name, hf_task_type]))

            dataset_name = parse_dataset(get_detailed_string([hf_dataset_type, hf_dataset_name]))

            split = detect_split(hf_dataset_split)
            if split == '':
                split = detect_split(get_detailed_string([hf_metric_type, hf_metric_name]))
            if split == '':
                split = 'test'

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
                                candidate_hf_metrics.append((k+' '+kk, str(vv)))
                        else:
                            candidate_hf_metrics.append((k, str(v)))
            else:
                candidate_hf_metrics = [(get_detailed_string([hf_metric_type, hf_metric_name]), str(hf_metric_value))]

            for mname, mvalue in candidate_hf_metrics:
                parsed_mname = parse_metric(mname)
                parsed_mname = mname if parsed_mname == '' else parsed_mname
                norm_mvalue = normalize_metric_value(parsed_mname, mvalue)
                metric_info = (mname, norm_mvalue)

                annotations.append(
                    dict(
                        task=task_name,
                        dataset=dataset_name,
                        metric=metric_info,
                    )
                )
    else:
        annotations = None
    
    return annotations


def get_manually_annotations():
    pass