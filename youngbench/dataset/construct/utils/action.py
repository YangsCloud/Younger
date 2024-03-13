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

from typing import Any, Generator

from .schema import Model, HFInfo

API_ADDRESS = 'https://api.yangs.cloud/'
YBD_MODEL_POINT = 'items/Young_Bench_Dataset_Model'
YBD_NETWORK_POINT = 'items/Young_Bench_Dataset_Network'
HFI_POINT = 'items/Hugging_Face_Info'


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


def read_limit_model_items(token: str, limit: int = 100, filter: dict | None = None, fields: list[str] | None = None) -> Generator[Model, None, None]:
    headers = get_headers(token)

    params = dict()
    if filter:
        params['filter'] = json.dumps(filter)

    if fields:
        params['fields'] = f'{fields[0]}'
        for field in fields[1:]:
            params['fields'] += f',{field}'

    params['limit'] = limit

    response = requests.get(API_ADDRESS+YBD_MODEL_POINT, headers=headers, params=params)
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


def read_model_items_by_model_id(model_id: str, token: str) -> list[Model]:
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


def read_model_items_manually(token: str, filter: dict | None = None, fields: list[str] | None = None) -> Generator[Model, None, None]:
    headers = get_headers(token)

    response = requests.get(API_ADDRESS+YBD_MODEL_POINT+'?aggregate[count]=*', headers=headers)

    data = response.json()
    count = data['data'][0]['count']
    limit = 100
    quotient, remainder = divmod(count, limit)
    pages = quotient + (remainder > 0)

    params = dict()
    if filter:
        params['filter'] = json.dumps(filter)

    if fields:
        params['fields'] = f'{fields[0]}'
        for field in fields[1:]:
            params['fields'] += f',{field}'

    params['limit'] = limit

    for page in range(1, pages+1):
        params['page'] = page
        response = requests.get(API_ADDRESS+YBD_MODEL_POINT, headers=headers, params=params)
        data = response.json()
        for d in data['data']:
            yield Model(**d)


def read_hfinfo_items_manually(token: str, filter: dict | None = None, fields: list[str] | None = None) -> Generator[HFInfo, None, None]:
    headers = get_headers(token)

    response = requests.get(API_ADDRESS+HFI_POINT+'?aggregate[count]=*', headers=headers)

    data = response.json()
    count = data['data'][0]['count']
    limit = 100
    quotient, remainder = divmod(count, limit)
    pages = quotient + (remainder > 0)

    params = dict()
    if filter:
        params['filter'] = json.dumps(filter)

    if fields:
        params['fields'] = f'{fields[0]}'
        for field in fields[1:]:
            params['fields'] += f',{field}'

    params['limit'] = limit

    for page in range(1, pages+1):
        params['page'] = page
        response = requests.get(API_ADDRESS+HFI_POINT, headers=headers, params=params)
        data = response.json()
        for d in data['data']:
            yield HFInfo(**d)


def update_model_item_by_model_id(model_id: str, model: Model, token: str) -> Model:
    headers = get_headers(token)

    model_items = read_model_items_by_model_id(model_id, token)
    updated_model_item = None
    if len(model_items) == 1:
        model_item_id = model_items[0].id
        item = model.dict()
        response = requests.patch(API_ADDRESS+YBD_MODEL_POINT+f'/{model_item_id}', headers=headers, json=item)
        data = response.json()
        if 'model_id' in data['data']:
            updated_model_item = Model(**data['data'])
    return updated_model_item


def update_model_items_by_model_ids(model_ids: str, data: dict[str, Any], token: str) -> Model:
    headers = get_headers(token)

    model_item_ids = list()
    for model_id in model_ids:
        filter = {
            'model_id': {
                '_eq': model_id
            }
        }
        model_items = list(read_model_items_manually(token, filter=filter, fields=['id']))
        model_item_ids.append(model_items[0].id)

    body = {
        'keys': model_item_ids,
        'data': data
    }
    response = requests.patch(API_ADDRESS+YBD_MODEL_POINT, headers=headers, json=body)
    data = response.json()
    updated_model_items = list()
    for d in data['data']:
        updated_model_items.append(Model(**d))
    return updated_model_items


# HFI
def create_hfinfo_item(hfinfo: HFInfo, token: str) -> HFInfo:
    headers = get_headers(token)
    item = hfinfo.dict()
    response = requests.post(API_ADDRESS+HFI_POINT, headers=headers, json=item)
    data = response.json()
    hfinfo_item = None
    if 'model_id' in data['data']:
        hfinfo_item = HFInfo(**data['data'])
    return hfinfo_item


def create_hfinfo_items(hfinfos: list[HFInfo], token: str) -> list[HFInfo]:
    headers = get_headers(token)
    items = [hfinfo.dict() for hfinfo in hfinfos]
    response = requests.post(API_ADDRESS+HFI_POINT, headers=headers, json=items)
    data = response.json()
    hfinfo_items = list()
    for d in data['data']:
        hfinfo_items.append(HFInfo(**d))
    return hfinfo_items