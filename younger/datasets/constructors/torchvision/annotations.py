#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-12 12:01
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from younger.datasets.constructors.torchvision.utils import get_torchvision_model_type

def get_heuristic_annotations(model_id: str, model_metrics: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, dict[str, str]]] | None:
    annotations = dict()
    annotations['datasets'] = None
    annotations['language'] = None
    annotations['base_model'] = None

    # model_metrics: {
    #     Model_Variation: 
    #     {
    #         Dataset_Split:
    #         {
    #             metric_name: metric_value
    #         }
    #     }
    # }
    task = get_torchvision_model_type(model_id)
    metric_names = set()
    annotations['eval_results'] = list()
    for variation, variation_metrics in model_metrics.items():
        for dataset_name, metrics in variation_metrics.items():
            metric_names = metric_names | set(metrics.keys())
            for metric_name, metric_value in metrics.items():
                # TODO: Keep The Same Format With HuggingFace
                if metric_name in {'acc@1', 'acc@5', 'miou', 'pixel_acc', 'box_map', 'kp_map'}:
                    metric_value = metric_value / 100
                if metric_name == 'acc@1':
                    metric_name = 'accuracy @1'
                if metric_name == 'acc@5':
                    metric_name = 'accuracy @5'
                annotations['eval_results'].append(
                    dict(
                        task=task,
                        dataset=(dataset_name, 'test'),
                        metric=(metric_name, metric_value),
                        variation=variation
                    )
                )

    annotations['metrics'] = list(metric_names)

    return annotations


def get_manually_annotations():
    pass