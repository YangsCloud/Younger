#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-13 09:19
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict


class Node(object):
    def __init__(
            self,
            operator_type: str,
            operator_domain: str,
            attributes: Dict,
            parameters: Dict,
            operands: Dict,
            results: Dict,
            is_first: bool = False,
            is_last: bool = False,
            is_custom: bool = False,
            has_subgraph: bool = False,
    ) -> None:
        self._type = operator_type
        self._domain = operator_domain

        self._attributes = attributes
        self._parameters = parameters
        self._operands = operands
        self._results = results
        self._is_first = is_first
        self._is_last = is_last

        self._is_custom = is_custom
        self._has_subgraph = has_subgraph

    @property
    def features(self) -> Dict:
        features = dict(
            type = self._type,
            domain = self._domain,
            in_number = len(self._operands),
            out_number = len(self._results),
            is_first = self._is_first,
            is_last = self._is_last,
            is_custom = self._is_custom,
            has_subgraph = self._has_subgraph,
        )
        return features

    @property
    def dict(self) -> Dict:
        return dict(
            operator_type = self._type,
            operator_domain = self._domain,
            attributes = self._attributes,
            parameters = self._parameters,
            operands = self._operands,
            results = self._results,
            is_first = self._is_first,
            is_last = self._is_last,
            is_custom = self._is_custom,
            has_subgraph = self._has_subgraph,
        )

    @property
    def quasi_dict(self) -> Dict:
        quasi_dict = self.dict
        if self.is_custom:
            if '__YBD_function__' in quasi_dict['attributes']:
                quasi_dict['attributes'] = dict(__YBD_function__='')
            else:
                quasi_dict['attributes'] = dict()
        else:
            for attribute_name, attribute_value in quasi_dict['attributes'].items():
                if isinstance(attribute_value, list):
                    quasi_dict['attributes'][attribute_name] = list()
        return quasi_dict

    @property
    def type(self) -> str:
        return self._type

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def attributes(self) -> Dict:
        return self._attributes

    @property
    def parameters(self) -> Dict:
        return self._parameters

    @property
    def operands(self) -> Dict:
        return self._operands

    @property
    def results(self) -> Dict:
        return self._results

    @property
    def is_first(self) -> bool:
        return self._is_first

    @property
    def is_last(self) -> bool:
        return self._is_last

    @property
    def is_custom(self) -> bool:
        return self._is_custom

    @property
    def has_subgraph(self) -> bool:
        return self._has_subgraph