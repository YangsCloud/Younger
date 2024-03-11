#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-11 21:01
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import requests

from typing import Generator

from .schema import Model

API_ADDRESS = 'https://api.yangs.cloud/'
YBD_MODEL_POINT = 'items/Young_Bench_Dataset_Model'
YBD_NETWORK_POINT = 'items/Young_Bench_Dataset_Network'


def get_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def create_model_item(model: Model, token: str) -> Model:
    headers = get_headers(token)
    item = model.dict()
    response = requests.post(API_ADDRESS+YBD_MODEL_POINT, headers=headers, json=item)
    data = response.json()
    model_item = None
    if 'model_id' in data['data']:
        model_item = Model(**data['data'])
    return model_item


def create_model_items(models: list[Model], token: str) -> list[Model]:
    headers = get_headers(token)
    items = [model.dict() for model in models]
    response = requests.post(API_ADDRESS+YBD_MODEL_POINT, headers=headers, json=items)
    data = response.json()
    model_items = list()
    for d in data['data']:
        model_items.append(Model(**d))
    return model_items


def read_model_items(token: str) -> Generator[Model, None, None]:
    headers = get_headers(token)
    response = requests.get(API_ADDRESS+YBD_MODEL_POINT+'?aggregate[count]=*', headers=headers)

    data = response.json()
    count = data['data'][0]['count']
    limit = 100
    quotient, remainder = divmod(count, limit)
    pages = quotient + (remainder > 0)

    for page in range(1, pages+1):
        response = requests.get(API_ADDRESS+YBD_MODEL_POINT, headers=headers, params=dict(limit=limit, page=page))
        data = response.json()
        for d in data['data']:
            yield Model(**d)


def read_model_item_by_model_id(model_id: str, token: str) -> list[Model]:
    headers = get_headers(token)
    filter = {
        'model_id': {
            "_eq": model_id
        }
    }
    params = dict(
        filter = json.dumps(filter)
    )
    response = requests.get(API_ADDRESS+YBD_MODEL_POINT, params=params, headers=headers)
    data = response.json()
    model_items = list()
    for d in data['data']:
        model_items.append(Model(**d))
    return model_items


def read_model_item_using_filter(filter: dict, token: str) -> list[Model]:
    headers = get_headers(token)
    params = dict(
        filter = json.dumps(filter)
    )
    response = requests.get(API_ADDRESS+YBD_MODEL_POINT, params=params, headers=headers)
    data = response.json()
    model_items = list()
    for d in data['data']:
        model_items.append(Model(**d))
    return model_items


def update_model_item_by_model_id(model_id: str, model: Model, token: str) -> Model:
    headers = get_headers(token)

    model_items = read_model_item_by_model_id(model_id, token)
    updated_model_item = None
    if len(model_items) == 1:
        model_item_id = model_items[0]['id']
        item = model.dict()
        response = requests.patch(API_ADDRESS+YBD_MODEL_POINT+f'/{model_item_id}', headers=headers, json=item)
        data = response.json()
        if 'model_id' in data['data']:
            updated_model_item = Model(**data['data'])
    return updated_model_item