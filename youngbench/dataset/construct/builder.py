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
import shutil
import pathlib
import argparse
import semantic_version

from huggingface_hub import login
from optimum.exporters.onnx import main_export
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from youngbench.dataset.modules import Dataset, Instance
from youngbench.dataset.modules import Instance

from youngbench.dataset.utils.io import load_model
from youngbench.dataset.utils.cache import set_cache_root, get_cache_root
from youngbench.dataset.construct.utils.get_model import cache_model
from youngbench.logging import set_logger, logger


MAX_SIZE = 4 * 1024 * 1024 * 1024


def get_directory_size(directory):
    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    return total_size


def get_opset_version(onnx_model: onnx.ModelProto):
    for opset_info in onnx_model.opset_import:
        if opset_info.domain == "":
            opset_version = opset_info.version
            break

    return opset_version


def get_instance_dirname(model_id: str, onnx_model_filename: str):
    model_name = model_id.replace('/', '--HF--')
    onnx_model_name = os.path.splitext(onnx_model_filename)[0]
    return model_name + '--YBDI--' + onnx_model_name


def convert_hf_onnx(model_id: str, output_dir: str, device: str = 'cpu', cache_dir: str | None = None) -> list[str]:
    assert device in {'cpu', 'cuda'}
    try:
        # meta_data = huggingface_hub.get_hf_file_metadata(huggingface_hub.hf_hub_url(repo_id="julien-c/EsperBERTo-small", filename="pytorch_model.bin"))
        # meta_data.size
        cache_model(model_id, cache_dir=cache_dir, monolith=False)
        infered_model_size = get_directory_size(cache_dir)
        if infered_model_size > MAX_SIZE:
            raise MemoryError(f'Model Size: {infered_model_size} Memory Maybe Occupy Too Much While Exporting')
        #main_export(model_id, output_dir, device=device, cache_dir=cache_dir, monolith=False, do_validation=False, trust_remote_code=True)
        main_export(model_id, output_dir, device=device, cache_dir=cache_dir, monolith=True, do_validation=False, trust_remote_code=True)
    except MemoryError as e:
        logger.error(f'Model ID = {model_id}: Skip! Maybe OOM - {e}')
    except Exception as e:
        logger.error(f'Model ID = {model_id}: Conversion Error - {e} ')

    onnx_model_filenames = list()
    for filename in os.listdir(output_dir):
        if os.path.splitext(filename)[1] == '.onnx':
            onnx_model_filenames.append(filename)

    return onnx_model_filenames


def clean_dir(dirpath: pathlib.Path):
    for filename in os.listdir(dirpath):
        filepath = dirpath.joinpath(filename)
        if filepath.is_dir():
            shutil.rmtree(filepath)
        if filepath.is_file():
            os.remove(filepath)


def clean_hfmodel_cache(model_id: str, cache_dir: str | None = None):
    repo_type = "model"
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    
    cache_dirpath = pathlib.Path(cache_dir)

    object_id = model_id.replace("/", "--")
    model_cache = cache_dirpath.joinpath(f"{repo_type}s--{object_id}")
    if model_cache.is_dir():
        clean_dir(model_cache)
        os.rmdir(model_cache)


def clean_convert_cache(convert_cache_dirpath: pathlib.Path = None):
    clean_dir(convert_cache_dirpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Instances of The Young Neural Network Architecture Dataset from HuggingFace Hub.")

    # Dataset Release Version.
    parser.add_argument('--version', type=str, required=True)

    # HuggingFace Hub Cache Dir
    parser.add_argument('--hf-cache-dirpath', type=str, default='')

    parser.add_argument('--convert-cache-dirpath', type=str, default='')

    # Model IDs File
    parser.add_argument('--model-id-filepath', type=str, default='')

    # Instances Save/Load Path.
    parser.add_argument('--save-dirpath', type=str, default='')
    parser.add_argument('--process-flag-path', type=str, default='./process.flg')

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--logging-path', type=str, default='')

    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    assert semantic_version.validate(args.version), f'The version provided must follow the SemVer 2.0.0 Specification.'
    version = semantic_version.Version(args.version)

    set_logger(path=args.logging_path)

    if args.hf_token is not None:
        login(token=args.hf_token)

    if args.hf_cache_dirpath == '':
        hf_cache_dirpath = None
    else:
        hf_cache_dirpath = pathlib.Path(args.hf_cache_dirpath)
        assert hf_cache_dirpath.is_dir(), f'Cache Directory Path Does Not Exists.'
        logger.info(f'HuggingFace Hub Cache directory: {hf_cache_dirpath.absolute()}')

    #model_ids = list()
    #with open(args.model_id_filepath, 'r') as f:
    #    for line in f:
    #        model_info = json.loads(line)
    #        model_ids.append(model_info['model_id'])

    with open(args.model_id_filepath, 'r') as f:
        model_ids = json.load(f)

    dataset = Dataset()

    save_dirpath = pathlib.Path(args.save_dirpath)
    #assert not save_dirpath.is_dir(), f'Directory exists at the specified \"Save Path\": {save_dirpath}.'

    logger.info(f'-> Dataset Initialized.')

    if args.convert_cache_dirpath != '':
        set_cache_root(pathlib.Path(args.convert_cache_dirpath))
    convert_cache_dirpath = get_cache_root()

    assert args.device in {'cpu', 'cuda'}
    device = args.device

    succ_flags = set()
    fail_flags = set()
    partial_fail_flags = dict()

    process_flag_path = pathlib.Path(args.process_flag_path)
    if process_flag_path.is_file():
        with open(process_flag_path, 'r') as f:
            for line in f:
                process_json = line.strip()
                if len(process_json) == 0:
                    continue
                process_flag: dict = json.loads(process_json)

                if process_flag['mode'] == 'succ': # {'record': model_id (str)}
                    succ_flags.add(process_flag['record'])
                if process_flag['mode'] == 'fail': # {'record': (model_id (str), list[onnx_model_filename (str)]) | model_id (str)}
                    record = process_flag['record']
                    if isinstance(record, tuple):
                        assert record[0] not in partial_fail_flags
                        partial_fail_flags[record[0]] = record[1]

                    if isinstance(record, str):
                        fail_flags.add(record)

    logger.info(f'Already Converted: {len(succ_flags)}')
    logger.info(f'Failed Converted: {len(fail_flags)}')

    logger.info(f'-> Instances Creating ...')
    overalli = 0
    h2o_succ = 0
    h2o_fail = 0
    for index, model_id in enumerate(model_ids, start=1):
        if model_id in succ_flags:
            logger.info(f' # Skip No.{index} model: {model_id}. Reason: Success Before.')
            continue
        if model_id in fail_flags:
            logger.info(f' # Skip No.{index} model: {model_id}. Reason: Failure Before.')
            continue
        if model_id in partial_fail_flags:
            logger.info(f' # Skip No.{index} model: {model_id}. Reason: [HF -> ONNX] Success But [ONNX -> NetworkX] Failure Before. Failure={len(partial_fail_flags[model_id])} ...')
            continue

        logger.info(f' # No.{index}: Now processing the model: {model_id} ...')
        logger.info(f' v - Converting Model into ONNX:')
        onnx_model_filenames = convert_hf_onnx(model_id, convert_cache_dirpath, device=device, cache_dir=hf_cache_dirpath)
        logger.info(f' ^ - Converted: Got {len(onnx_model_filenames)} ONNX Models.')
        if len(onnx_model_filenames) != 0:
            h2o_succ += 1
            mode = 'succ'
        else:
            h2o_fail += 1
            mode = 'fail'

        o2n_succ = 0
        o2n_fail = 0
        o2n_fail_flags = list()
        for onnx_model_filename in onnx_model_filenames:
            onnx_model_filepath = convert_cache_dirpath.joinpath(onnx_model_filename)
            # onnx_model = load_model(onnx_model_filepath)
            # opset = get_opset_version(onnx_model)

            logger.info(f'      > Converting ONNX -> NetworkX: ONNX Filepath - {onnx_model_filepath}')
            # logger.info(f'      > Converting ONNX -> NetworkX: ONNX Filepath - {onnx_model_filepath} (ONNX opset={opset})')
            try:
                instance = Instance(model=onnx_model_filepath, labels=dict(source='HuggingFace', model_id=model_id, onnx_model_filename=onnx_model_filename))
                instance_dirpath = save_dirpath.joinpath(get_instance_dirname(model_id, onnx_model_filename))
                instance.save(instance_dirpath)
                o2n_succ += 1
                overalli += 1
                logger.info(f'        No.{overalli} Instance Saved: {instance_dirpath}')
            except Exception as e:
                logger.error(f'Error! [ONNX -> NetworkX Error] OR [Dataset Insertion Error] - {e}')
                o2n_fail += 1
                o2n_fail_flags.append(onnx_model_filename)

            logger.info(f'      > Converted. Succ/Fail/Total=({o2n_succ}/{o2n_fail}/{len(onnx_model_filenames)})')

        with open(process_flag_path, 'a') as f:
            h2o_process_flag = json.dumps(dict(mode=mode, record=model_id))
            f.write(f'{h2o_process_flag}\n')
            if len(o2n_fail_flags) != 0:
                o2n_process_flag = json.dumps(dict(mode='fail', record=(model_id, o2n_fail_flags)))
                f.write(f'{o2n_process_flag}\n')

        clean_hfmodel_cache(model_id, cache_dir=hf_cache_dirpath)
        clean_convert_cache(convert_cache_dirpath)
        logger.info(f' = # No.{index}: Finished. Succ/Fail/Total=({h2o_succ}/{h2o_fail}/{len(model_ids)})')

    logger.info(f'-> Instances Created.')
