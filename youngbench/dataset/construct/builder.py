#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-02 13:57
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import onnx
import pathlib
import argparse
import semantic_version

from typing import Generator, Tuple
from optimum.exporters.onnx import main_export

from youngbench.dataset.modules import Dataset
from youngbench.dataset.modules import Instance

from youngbench.dataset.utils.io import load_model
from youngbench.dataset.utils.cache import set_cache_root, get_cache_root
from youngbench.logging import set_logger, logger


def get_opset_version(onnx_model: onnx.ModelProto):
    for opset_info in onnx_model.opset_import:
        if opset_info.domain == "":
            opset_version = opset_info.version
            break

    return opset_version


def convert_hf_onnx(model_id: str, output_dir: str, device: str = 'cpu', cache_dir: str | None = None) -> list[str]:
    assert device in {'cpu', 'cuda'}
    try:
        main_export(model_id, output_dir, device=device, cache_dir=cache_dir)
    except Exception as e:
        logger.error(f'Model ID = {model_id}: Conversion Error - {e} ')

    onnx_model_filenames = list()
    for filename in os.listdir(output_dir):
        if os.path.splitext(filename)[1] == '.onnx':
            onnx_model_filenames.append(filename)

    return onnx_model_filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/Update The Young Neural Network Architecture Dataset HuggingFace Hub (YoungBench - Dataset).")

    # Dataset Release Version.
    parser.add_argument('--version', type=str, required=True)

    # HuggingFace Hub Cache Dir
    parser.add_argument('--hf-cache-dirpath', type=str, default='')

    parser.add_argument('--convert-cache-dirpath', type=str, default='')

    # Model IDs File
    parser.add_argument('--model-id-filepath', type=str, default='')

    # Dataset Save/Load Path.
    parser.add_argument('--load-dirpath', type=str, default='')
    parser.add_argument('--save-dirpath', type=str, default='')
    parser.add_argument('--process-flag-path', type=str, default='./process.flg')


    parser.add_argument('--logging-path', type=str, default='')

    parser.add_argument('--device', type=str, default='cpu')

    # Mode - Create / Update
    parser.add_argument('--mode', type=str, default='Update', choices=['Update', 'Create'])

    args = parser.parse_args()

    assert semantic_version.validate(args.version), f'The version provided must follow the SemVer 2.0.0 Specification.'
    version = semantic_version.Version(args.version)

    set_logger(path=args.logging_path)

    if args.hf_cache_dirpath == '':
        hf_cache_dirpath = None
    else:
        hf_cache_dirpath = pathlib.Path(args.hf_cache_dirpath)
        assert hf_cache_dirpath.is_dir(), f'Cache Directory Path Does Not Exists.'
        logger.info(f'HuggingFace Hub Cache directory: {hf_cache_dirpath.absolute()}')

    model_ids = list()
    with open(args.model_id_filepath, 'r') as f:
        for line in f:
            model_info = json.loads(line)
            model_ids.append(model_info['model_id'])

    dataset = Dataset()

    if args.mode == 'Create':
        save_dirpath = pathlib.Path(args.save_dirpath)
        assert not save_dirpath.is_dir(), f'Directory exists at the specified \"Save Path\": {save_dirpath}.'

    if args.mode == 'Update':
        if args.load_dirpath == '':
            load_dirpath = pathlib.Path('.')
        else:
            load_dirpath = pathlib.Path(args.load_dirpath)
        assert load_dirpath.is_dir(), f'Directory does not exists at the specified \"Load Path\": {load_dirpath}.'

        if args.save_dirpath == '':
            save_dirpath = load_dirpath
        else:
            save_dirpath = pathlib.Path(args.save_dirpath)
        assert not save_dirpath.is_dir(), f'Directory exists at the specified \"Save Path\": {save_dirpath}.'

        dataset.load(load_dirpath)

    logger.info(f'-> Dataset Initialized.')

    if args.convert_cache_dirpath != '':
        set_cache_root(pathlib.Path(args.convert_cache_dirpath))
    convert_cache_dirpath = get_cache_root()
    
    assert args.device in {'cpu', 'cuda'}
    device = args.device

    succ_flags = list()
    fail_flags = list()

    process_flags = set()
    process_flag_path = pathlib.Path(args.process_flag_path)
    if process_flag_path.is_file():
        with open(process_flag_path, 'r') as f:
            for line in f:
                process_json = line.strip()
                if len(process_json) == 0:
                    continue
                process_flag: dict = json.loads(process_json)

                if process_flag['mode'] == 'succ':
                    succ_flags.append(process_flag['record'])
                if process_flag['mode'] == 'fail':
                    fail_flags.append(process_flag['record']) # {'record': list[(model_id, onnx_model_filename)] | model_id }

    logger.info(f'Already Converted: {len(succ_flags)}')
    logger.info(f'Failed Converted: {len(fail_flags)}')

    logger.info(f'-> Dataset Creating ...')
    h2o_succ = 0
    h2o_fail = 0
    for index, model_id in enumerate(model_ids, start=1):
        logger.info(f' # {index}: Now processing the model: {model_id} ...')
        logger.info(f' v - Converting Model into ONNX:')
        onnx_model_filenames = convert_hf_onnx(model_id, convert_cache_dirpath, device=device, cache_dir=hf_cache_dirpath)
        logger.info(f' ^ - Converted: Got {len(onnx_model_filenames)} ONNX Models.')
        if len(onnx_model_filenames) != 0:
            h2o_succ += 1
            mode = 'succ'
        else:
            h2o_fail += 1
            mode = 'fail'

        with open(process_flag_path, 'a') as f:
            process_flag = json.dumps(dict(mode=mode, record=model_id))
            f.write(f'{process_flag}\n')

        o2n_succ = 0
        o2n_fail = 0
        o2n_fail_flags = list()
        for onnx_model_filename in onnx_model_filenames:
            onnx_model_filepath = convert_cache_dirpath.joinpath(onnx_model_filename)
            onnx_model = load_model(onnx_model_filepath)
            opset = get_opset_version(onnx_model)

            logger.info(f'      > Converting ONNX -> NetworkX: ONNX Filepath - {onnx_model_filepath} (ONNX opset={opset})')
            try:
                dataset.insert(Instance(model=onnx_model, labels=dict(source='HuggingFace', model_id=model_id, name=onnx_model_filename)))
                o2n_succ += 1
            except Exception as e:
                logger.error(f'Error! [ONNX -> NetworkX Error] OR [Dataset Insertion Error] - {e}')
                o2n_fail += 1
                o2n_fail_flags.append((model_id, onnx_model_filename))

            logger.info(f'      > Converted. Succ/Fail/Total=({o2n_succ}/{o2n_fail}/{len(onnx_model_filenames)})')

        with open(process_flag_path, 'a') as f:
            process_flag = json.dumps(dict(mode=mode, record=o2n_fail_flags))
            f.write(f'{process_flag}\n')

        logger.info(f' !# {index}: Finished. Succ/Fail/Total=({h2o_succ}/{h2o_fail}/{len(model_ids)})')

    logger.info(f'-> Created.')

    logger.info(f'-> Dataset Releasing ...')
    dataset.release(version)
    logger.info(f'-> Released.')

    logger.info(f'-> Dataset Saving ...')
    dataset.save(save_dirpath)
    logger.info(f'-> Saved: {save_dirpath}.')