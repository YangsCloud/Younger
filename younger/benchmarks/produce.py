#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-07-15 15:46
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

from younger.commons.io import load_json, save_json
from younger.commons.logging import logger

from younger.benchmarks.utils import get_instances, get_op_string


def main(real_dirpath: pathlib.Path, product_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path | None = None):
    logger.info(f' v Loading all Instances in Younger ...')
    instances = get_instances(real_dirpath, remove_tiny=30)
    logger.info(f' ^ Done')

    if statistics_dirpath is None:
        logger.info(f' - Statistics are not specified, skip ...')
        statistics = {'op_frequency': dict()}
    else:
        logger.info(f' - Statistics are specified')
        logger.info(f' v Loading Statistics ...')
        statistics = load_json(statistics_dirpath.joinpath('statistics.json'))
        logger.info(f' ^ Done')
    compare_op_frequency = statistics['op_frequency']

    logger.info(f' v Analyzing Younger ...')
    younger_statistics = dict()
    unknown_op_frequency: list[dict[str, int]] = list()
    younger_op_frequency: list[dict[str, int]] = list()
    overall_op_frequency: dict[str, int] = dict()
    for instance in instances:
        instance_op_frequency = dict()
        instance_unknown_op_frequency = dict()
        for node_index in instance.network.graph.nodes():
            op = get_op_string(instance.network.graph.nodes[node_index]['operator']['op_type'], instance.network.graph.nodes[node_index]['operator']['domain'])
            overall_op_frequency[op] = overall_op_frequency.get(op, 0) + 1
            instance_op_frequency[op] = instance_op_frequency.get(op, 0) + 1
            if op in compare_op_frequency:
                continue
            instance_unknown_op_frequency[op] = instance_unknown_op_frequency.get(op, 0) + 1
        younger_op_frequency.append(instance_op_frequency)
        unknown_op_frequency.append(instance_unknown_op_frequency)
    
    younger_statistics['overall_op_frequency'] = overall_op_frequency
    younger_statistics['younger_op_frequency'] = younger_op_frequency
    younger_statistics['unknown_op_frequency'] = unknown_op_frequency

    unseen_op_kind: list[int] = list()
    unseen_op_freq: list[int] = list()
    total = 0
    for instance_unknown_op_frequency in unknown_op_frequency:
        op_kind = 0
        op_freq = 0
        has_uns = False
        for op, frequency in instance_unknown_op_frequency.items():
            has_uns = True
            op_kind += 1
            op_freq += frequency
        total += has_uns
        unseen_op_kind.append(op_kind)
        unseen_op_freq.append(op_freq)
    
    younger_statistics['unseen_op_kind'] = {f'{instances[index].labels["hash"]}': op_kind for index, op_kind in enumerate(unseen_op_kind)}
    younger_statistics['unseen_op_freq'] = {f'{instances[index].labels["hash"]}': op_freq for index, op_freq in enumerate(unseen_op_freq)}

    younger_statistics_filepath = product_dirpath.joinpath('statistics.json')
    save_json(younger_statistics, younger_statistics_filepath, indent=2)
    logger.info(f'   Younger\'s JSON Statistics results saved into: {younger_statistics_filepath}')
    logger.info(f'   Total {total} instances have unseen operators.')
    logger.info(f' ^ Done')

    logger.info(f' v Producing possible instances ...')
    detailed_product: list[list[str]] = [
        {
            'kind': unseen_op_kind[index],
            'freq': unseen_op_freq[index],
            'overall_kind': len(younger_op_frequency[index].items()),
            'overall_freq': sum([freq for _, freq in younger_op_frequency[index].items()]),
            'detail': sorted([[op, freq] for op, freq in unknown_op_frequency[index].items()], key=lambda x: x[1], reverse=True),
            'model_name': instances[index].labels['model_name'],
        }
        for index, instance in enumerate(instances)
    ]

    no_quantize_product = list()
    for product in detailed_product:
        invalid = False
        for op, _ in product['detail']:
            if ('Integer' in op) or ('Quantize' in op):
                invalid = True
                break
        if not invalid:
            no_quantize_product.append(product)

    detailed_product = no_quantize_product

    kind_base_product = sorted(detailed_product, key=lambda x: x['kind'], reverse=True)
    freq_base_product = sorted(detailed_product, key=lambda x: x['freq'], reverse=True)

    logger.info(
        f'\n   Kind-Base:\n'
        f'    - Top1 : {kind_base_product[0]["kind"]}/{kind_base_product[0]["overall_kind"]} ({kind_base_product[0]["kind"]/kind_base_product[0]["overall_kind"]*100:.2f}%)\n'
        f'    - Top3 : {kind_base_product[2]["kind"]}/{kind_base_product[2]["overall_kind"]} ({kind_base_product[2]["kind"]/kind_base_product[2]["overall_kind"]*100:.2f}%)\n'
        f'    - Top5 : {kind_base_product[4]["kind"]}/{kind_base_product[4]["overall_kind"]} ({kind_base_product[4]["kind"]/kind_base_product[4]["overall_kind"]*100:.2f}%)\n'
        f'    - Top10: {kind_base_product[9]["kind"]}/{kind_base_product[9]["overall_kind"]} ({kind_base_product[9]["kind"]/kind_base_product[9]["overall_kind"]*100:.2f}%)\n'
    )

    logger.info(
        f'\n   Freq-Base:\n'
        f'    - Top1 : {freq_base_product[0]["freq"]}/{freq_base_product[0]["overall_freq"]} ({freq_base_product[0]["freq"]/freq_base_product[0]["overall_freq"]*100:.2f}%)\n'
        f'    - Top3 : {freq_base_product[2]["freq"]}/{freq_base_product[2]["overall_freq"]} ({freq_base_product[2]["freq"]/freq_base_product[2]["overall_freq"]*100:.2f}%)\n'
        f'    - Top5 : {freq_base_product[4]["freq"]}/{freq_base_product[4]["overall_freq"]} ({freq_base_product[4]["freq"]/freq_base_product[4]["overall_freq"]*100:.2f}%)\n'
        f'    - Top10: {freq_base_product[9]["freq"]}/{freq_base_product[9]["overall_freq"]} ({freq_base_product[9]["freq"]/freq_base_product[9]["overall_freq"]*100:.2f}%)\n'
    )

    kind_pct_base_product = sorted(detailed_product, key=lambda x: (x['kind']/x['overall_kind']) if x['overall_kind'] !=0 else 0, reverse=True)
    freq_pct_base_product = sorted(detailed_product, key=lambda x: (x['freq']/x['overall_freq']) if x['overall_freq'] !=0 else 0, reverse=True)

    logger.info(
        f'\n   Kind-Base:\n'
        f'    - Top1 : {kind_pct_base_product[0]["kind"]}/{kind_pct_base_product[0]["overall_kind"]} ({kind_pct_base_product[0]["kind"]/kind_pct_base_product[0]["overall_kind"]*100:.2f}%)\n'
        f'    - Top3 : {kind_pct_base_product[2]["kind"]}/{kind_pct_base_product[2]["overall_kind"]} ({kind_pct_base_product[2]["kind"]/kind_pct_base_product[2]["overall_kind"]*100:.2f}%)\n'
        f'    - Top5 : {kind_pct_base_product[4]["kind"]}/{kind_pct_base_product[4]["overall_kind"]} ({kind_pct_base_product[4]["kind"]/kind_pct_base_product[4]["overall_kind"]*100:.2f}%)\n'
        f'    - Top10: {kind_pct_base_product[9]["kind"]}/{kind_pct_base_product[9]["overall_kind"]} ({kind_pct_base_product[9]["kind"]/kind_pct_base_product[9]["overall_kind"]*100:.2f}%)\n'
    )

    logger.info(
        f'\n   Freq-Base:\n'
        f'    - Top1 : {freq_pct_base_product[0]["freq"]}/{freq_pct_base_product[0]["overall_freq"]} ({freq_pct_base_product[0]["freq"]/freq_pct_base_product[0]["overall_freq"]*100:.2f}%)\n'
        f'    - Top3 : {freq_pct_base_product[2]["freq"]}/{freq_pct_base_product[2]["overall_freq"]} ({freq_pct_base_product[2]["freq"]/freq_pct_base_product[2]["overall_freq"]*100:.2f}%)\n'
        f'    - Top5 : {freq_pct_base_product[4]["freq"]}/{freq_pct_base_product[4]["overall_freq"]} ({freq_pct_base_product[4]["freq"]/freq_pct_base_product[4]["overall_freq"]*100:.2f}%)\n'
        f'    - Top10: {freq_pct_base_product[9]["freq"]}/{freq_pct_base_product[9]["overall_freq"]} ({freq_pct_base_product[9]["freq"]/freq_pct_base_product[9]["overall_freq"]*100:.2f}%)\n'
    )

    kind_base_product_filepath = product_dirpath.joinpath('kind_base_products.json')
    save_json(kind_base_product, kind_base_product_filepath, indent=2)
    logger.info(f'   Kind Base Products saved into: {kind_base_product_filepath}')

    kind_pct_base_product_filepath = product_dirpath.joinpath('kind_pct_base_products.json')
    save_json(kind_pct_base_product, kind_pct_base_product_filepath, indent=2)
    logger.info(f'   Kind Percent Base Products saved into: {kind_pct_base_product_filepath}')

    freq_base_product_filepath = product_dirpath.joinpath('freq_base_products.json')
    save_json(freq_base_product, freq_base_product_filepath, indent=2)
    logger.info(f'   Freq Base Products saved into: {freq_base_product_filepath}')

    freq_pct_base_product_filepath = product_dirpath.joinpath('freq_pct_base_products.json')
    save_json(freq_pct_base_product, freq_pct_base_product_filepath, indent=2)
    logger.info(f'   Freq Percent Base Products saved into: {freq_pct_base_product_filepath}')

    logger.info(f' ^ Done')