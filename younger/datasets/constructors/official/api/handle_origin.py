#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 22:23
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import json
import requests

from typing import Generator

from younger.datasets.utils.constants import YoungerAPI

from younger.datasets.constructors.official.api.schema import Model


ORIGIN_PREFIX = YoungerAPI.API_ADDRESS + YoungerAPI.ORIGIN_POINT


def get_headers(token: str):
    return {
        "Authorization": f"Bearer {token}"
    }


# v CRUD

def read_origin_size(token: str) -> int:
    headers = get_headers(token)
    response = requests.get(ORIGIN_PREFIX + '?aggregate[count]=*', headers=headers)
    data = response.json()
    return data['data'][0]['count']


def create_origin_items(models: list[Model], token: str) -> list[Model]:
    headers = get_headers(token)
    items = [model.dict() for model in models]
    response = requests.post(ORIGIN_PREFIX, headers=headers, json=items)
    data = response.json()
    model_items = list()
    for d in data['data']:
        model_items.append(Model(**d))
    return model_items


def read_origin_items(token: str, count: int | None = None, filter: dict | None = None, fields: list[str] | None = None, limit: int = 100) -> Generator[Model, None, None]:
    headers = get_headers(token)

    count = count if count else read_origin_size(token)

    params = dict()
    if filter:
        params['filter'] = json.dumps(filter)

    if fields:
        params['fields'] = f'{fields[0]}'
        for field in fields[1:]:
            params['fields'] += f',{field}'

    limit = 100
    quotient, remainder = divmod(count, limit)
    pages = quotient + (remainder > 0)

    params['limit'] = limit

    for page in range(1, pages+1):
        params['page'] = page
        response = requests.get(ORIGIN_PREFIX, headers=headers, params=params)
        data = response.json()
        for d in data['data']:
            yield Model(**d)