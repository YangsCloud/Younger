#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-11 20:08
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import json
import copy
import torch
import inspect
import importlib
import dataclasses
import huggingface_hub

from functools import partial
from typing import List, Dict, Type, Union, Tuple, Optional
from packaging import version
from urllib.parse import urlsplit
from requests.exceptions import ConnectionError as RequestsConnectionError

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from optimum.exporters.tasks import TasksManager
from optimum.configuration_utils import _transformers_version
from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED

from timm import __version__ as timm_version
from timm.models import load_model_config_from_hf, is_model
from timm.layers import set_layer_config
from timm.models._pretrained import PretrainedCfg
from timm.models._builder import _resolve_pretrained_source
from timm.models._hub import _get_safe_alternatives

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from transformers import AutoConfig, PretrainedConfig, BitsAndBytesConfig, PreTrainedModel, TFPreTrainedModel, is_torch_available, is_tf_available
from transformers.utils import cached_file, CONFIG_NAME, extract_commit_hash, is_peft_available, find_adapter_config_file, is_safetensors_available, TF_WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF2_WEIGHTS_INDEX_NAME, FLAX_WEIGHTS_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME, has_file
from transformers.utils.hub import get_checkpoint_shard_files
from transformers.modeling_utils import _add_variant
from transformers.safetensors_conversion import auto_conversion

from sentence_transformers.models import Transformer, Normalize
from sentence_transformers.util import is_sentence_transformer_model, load_dir_path, load_file_path, import_from_string
from sentence_transformers import __version__ as stfs_version

from youngbench.logging import set_logger, logger

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False


HF_WEIGHTS_NAME = "pytorch_model.bin"  # default pytorch pkl
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"

hf_hub_download = partial(hf_hub_download, library_name="timm", library_version=timm_version)


def timm_hf_split(hf_id: str):
    # FIXME I may change @ -> # and be parsed as fragment in a URI model name scheme
    rev_split = hf_id.split('@')
    assert 0 < len(rev_split) <= 2, 'hf_hub id should only contain one @ character to identify revision.'
    hf_model_id = rev_split[0]
    hf_revision = rev_split[-1] if len(rev_split) > 1 else None
    return hf_model_id, hf_revision


def timm_load_state_dict_from_hf(model_id: str, filename: str = HF_WEIGHTS_NAME):
    # Modified from sources: timm.load_state_dict_from_hf
    hf_model_id, hf_revision = timm_hf_split(model_id)

    # Look for .safetensors alternatives and load from it if it exists
    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                cached_safe_file = hf_hub_download(repo_id=hf_model_id, filename=safe_filename, revision=hf_revision)
                logger.info(
                    f"[{model_id}] Safe alternative available for '{filename}' "
                    f"(as '{safe_filename}'). Loading weights using safetensors.")
                return
            except EntryNotFoundError:
                pass

    # Otherwise, load using pytorch.load
    hf_hub_download(hf_model_id, filename=filename, revision=hf_revision)
    logger.debug(f"[{model_id}] Safe alternative not found for '{filename}'. Loading weights using default pytorch.")
    return


def timm_parse_model_name(model_name: str):
    # Modified from sources: timm.parse_model_name
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme == 'hf-hub'
    return (parsed.scheme, parsed.path)


def timm_cache_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg = None,
    pretrained_cfg_overlay = None,
    scriptable: Optional[bool] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    **kwargs,
):
    # Modified from sources: timm.create_model & timm.models._builder.build_model_with_cfg

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = timm_parse_model_name(model_name)
    assert model_source == 'hf-hub'
    assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
    # For model names specified in the form `hf-hub:path/architecture_name@revision`,
    # load model weights + pretrained_cfg from Hugging Face hub.
    pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_name)
    print(model_name)
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        assert isinstance(pretrained_cfg, dict)
        pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        pretrained_cfg_overlay = pretrained_cfg_overlay or {}
        pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)
        pretrained_cfg = pretrained_cfg.to_dict()
        load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
        assert load_from == 'hf-hub'
        logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            timm_load_state_dict_from_hf(*pretrained_loc)
        else:
            timm_load_state_dict_from_hf(pretrained_loc)

    return


def get_transformer_model(
    model_id: str,
    max_seq_length: Optional[int] = None,
    model_args: Dict = {},
    cache_dir: Optional[str] = None,
    tokenizer_args: Dict = {},
    do_lower_case: bool = False,
    tokenizer_name_or_path: str = None,
):
    config = AutoConfig.from_pretrained(model_id, **model_args, cache_dir=cache_dir)
    fs_transformers_cache_model("AutoModel", model_id, config=config, cache_dir=cache_dir, **model_args)


def stfs_load_auto_model(
    model_id: str,
    token: Optional[Union[bool, str]],
    cache_folder: Optional[str],
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
):
    # Modified from sources: sentence_transformers.Transformer.__init__
    """
    Creates a simple Transformer + Mean Pooling model and returns the modules
    """
    logger.warning(
        "No sentence-transformers model found with name {}. Creating a new one with MEAN pooling.".format(
            id
        )
    )

    get_transformer_model(
        model_id,
        cache_dir=cache_folder,
        model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
        tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
    )

    return

def stfs_load_sbert_model(
    model_id: str,
    token: Optional[Union[bool, str]],
    cache_folder: Optional[str],
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """
    Loads a full sentence-transformers model
    """
    # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
    config_sentence_transformers_json_path = load_file_path(
        model_id,
        "config_sentence_transformers.json",
        token=token,
        cache_folder=cache_folder,
        revision=revision,
    )
    if config_sentence_transformers_json_path is not None:
        with open(config_sentence_transformers_json_path) as fIn:
            model_config = json.load(fIn)

        if (
            "__version__" in model_config
            and "sentence_transformers" in model_config["__version__"]
            and model_config["__version__"]["sentence_transformers"] > stfs_version
        ):
            logger.warning(
                "You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(
                    model_config["__version__"]["sentence_transformers"], stfs_version
                )
            )

    # Load the modules of sentence transformer
    modules_json_path = load_file_path(
        model_id, "modules.json", token=token, cache_folder=cache_folder, revision=revision
    )
    with open(modules_json_path) as fIn:
        modules_config = json.load(fIn)

    for module_config in modules_config:
        module_class = import_from_string(module_config["type"])
        # For Transformer, don't load the full directory, rely on `transformers` instead
        # But, do load the config file first.
        if module_class == Transformer and module_config["path"] == "":
            kwargs = {}
            for config_name in [
                "sentence_bert_config.json",
                "sentence_roberta_config.json",
                "sentence_distilbert_config.json",
                "sentence_camembert_config.json",
                "sentence_albert_config.json",
                "sentence_xlm-roberta_config.json",
                "sentence_xlnet_config.json",
            ]:
                config_path = load_file_path(
                    model_id, config_name, token=token, cache_folder=cache_folder, revision=revision
                )
                if config_path is not None:
                    with open(config_path) as fIn:
                        kwargs = json.load(fIn)
                    break
            hub_kwargs = {"token": token, "trust_remote_code": trust_remote_code, "revision": revision}
            if "model_args" in kwargs:
                kwargs["model_args"].update(hub_kwargs)
            else:
                kwargs["model_args"] = hub_kwargs
            if "tokenizer_args" in kwargs:
                kwargs["tokenizer_args"].update(hub_kwargs)
            else:
                kwargs["tokenizer_args"] = hub_kwargs
            get_transformer_model(model_id, cache_dir=cache_folder, **kwargs)
        else:
            # Normalize does not require any files to be loaded
            if module_class != Normalize:
                load_dir_path(
                    model_id,
                    module_config["path"],
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                )

    return


def stfs_cache_model(model_id, cache_folder=None, use_auth_token=None):
    token = use_auth_token
    if cache_folder is None:
        cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    if model_id is not None and model_id != "":
        logger.info("Load pretrained SentenceTransformer: {}".format(model_id))

        # Old models that don't belong to any organization
        basic_transformer_models = [
            "albert-base-v1",
            "albert-base-v2",
            "albert-large-v1",
            "albert-large-v2",
            "albert-xlarge-v1",
            "albert-xlarge-v2",
            "albert-xxlarge-v1",
            "albert-xxlarge-v2",
            "bert-base-cased-finetuned-mrpc",
            "bert-base-cased",
            "bert-base-chinese",
            "bert-base-german-cased",
            "bert-base-german-dbmdz-cased",
            "bert-base-german-dbmdz-uncased",
            "bert-base-multilingual-cased",
            "bert-base-multilingual-uncased",
            "bert-base-uncased",
            "bert-large-cased-whole-word-masking-finetuned-squad",
            "bert-large-cased-whole-word-masking",
            "bert-large-cased",
            "bert-large-uncased-whole-word-masking-finetuned-squad",
            "bert-large-uncased-whole-word-masking",
            "bert-large-uncased",
            "camembert-base",
            "ctrl",
            "distilbert-base-cased-distilled-squad",
            "distilbert-base-cased",
            "distilbert-base-german-cased",
            "distilbert-base-multilingual-cased",
            "distilbert-base-uncased-distilled-squad",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "distilbert-base-uncased",
            "distilgpt2",
            "distilroberta-base",
            "gpt2-large",
            "gpt2-medium",
            "gpt2-xl",
            "gpt2",
            "openai-gpt",
            "roberta-base-openai-detector",
            "roberta-base",
            "roberta-large-mnli",
            "roberta-large-openai-detector",
            "roberta-large",
            "t5-11b",
            "t5-3b",
            "t5-base",
            "t5-large",
            "t5-small",
            "transfo-xl-wt103",
            "xlm-clm-ende-1024",
            "xlm-clm-enfr-1024",
            "xlm-mlm-100-1280",
            "xlm-mlm-17-1280",
            "xlm-mlm-en-2048",
            "xlm-mlm-ende-1024",
            "xlm-mlm-enfr-1024",
            "xlm-mlm-enro-1024",
            "xlm-mlm-tlm-xnli15-1024",
            "xlm-mlm-xnli15-1024",
            "xlm-roberta-base",
            "xlm-roberta-large-finetuned-conll02-dutch",
            "xlm-roberta-large-finetuned-conll02-spanish",
            "xlm-roberta-large-finetuned-conll03-english",
            "xlm-roberta-large-finetuned-conll03-german",
            "xlm-roberta-large",
            "xlnet-base-cased",
            "xlnet-large-cased",
        ]
    if not os.path.exists(model_id):
    # Not a path, load from hub
        if "\\" in model_id or model_id.count("/") > 1:
            raise ValueError("Path {} not found".format(model_id))

        if "/" not in model_id and model_id.lower() not in basic_transformer_models:
            # A model from sentence-transformers
            model_id = __MODEL_HUB_ORGANIZATION__ + "/" + model_id

    if is_sentence_transformer_model(model_id, token, cache_folder=cache_folder, revision=None):
        stfs_load_sbert_model(
            model_id,
            token=token,
            cache_folder=cache_folder,
            revision=None,
            trust_remote_code=False,
        )
    else:
        stfs_load_auto_model(
            model_id,
            token=token,
            cache_folder=cache_folder,
            revision=None,
            trust_remote_code=False,
        )


def fs_cache_model(library_name, model_class_names, model_id, **kwargs):
    # Diffusers & Transformers Cache Model

    # Modified from sources: diffusers.pipelines.pipeline_utils.DiffusionPipeline & transformers.modeling_utils.PreTrainedModel.from_pretrained
    if library_name == "diffusers":
        assert len(model_class_names) == 1
        fs_diffusers_cache_model(model_class_names[0], model_id, **kwargs)
    if library_name == "transformers":
        for model_class_name in model_class_names:
            fs_transformers_cache_model(model_class_name, model_id, **kwargs)


def fs_diffusers_cache_model(model_class_name, model_id, **kwargs):
    assert model_class_name in {"StableDiffusionPipeline", "StableDiffusionXLImg2ImgPipeline"}
    cache_dir = kwargs.pop("cache_dir", None)
    resume_download = kwargs.pop("resume_download", False)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    from_flax = kwargs.pop("from_flax", False)
    custom_pipeline = kwargs.pop("custom_pipeline", None)
    custom_revision = kwargs.pop("custom_revision", None)
    variant = kwargs.pop("variant", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    use_onnx = kwargs.pop("use_onnx", None)
    load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)

    assert not os.path.isdir(model_id)
    if model_id.count("/") > 1:
        raise ValueError(
            f'The provided pretrained_model_name_or_path "{model_id}"'
            " is neither a valid local path nor a valid repo id. Please check the parameter."
        )
    DiffusionPipeline.download(
        model_id,
        cache_dir=cache_dir,
        resume_download=resume_download,
        force_download=force_download,
        proxies=proxies,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        from_flax=from_flax,
        use_safetensors=use_safetensors,
        use_onnx=use_onnx,
        custom_pipeline=custom_pipeline,
        custom_revision=custom_revision,
        variant=variant,
        load_connected_pipeline=load_connected_pipeline,
        **kwargs,
    )


def fs_transformers_cache_model(model_class_name: str, model_id, **kwargs):
    assert model_class_name.startswith("AutoModel") or model_class_name.startswith("TFAutoModel")
    if model_class_name.startswith("AutoModel"):
        fs_pt_tfs_cache_model(model_id, **kwargs)
    if model_class_name.startswith("TFAutoModel"):
        fs_tf_tfs_cache_model(model_id, **kwargs)


def fs_pt_tfs_cache_model(
    model_id,
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    revision = "main",
    force_download = False,
    local_files_only = False,
    token = None,
    use_safetensors = None,
    **kwargs
):
    from_tf = kwargs.pop("from_tf", False)
    from_flax = kwargs.pop("from_flax", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    use_auth_token = kwargs.pop("use_auth_token", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    _ = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    load_in_8bit = kwargs.pop("load_in_8bit", False)
    load_in_4bit = kwargs.pop("load_in_4bit", False)
    quantization_config = kwargs.pop("quantization_config", None)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    variant = kwargs.pop("variant", None)
    adapter_kwargs = kwargs.pop("adapter_kwargs", {})

    if use_auth_token is not None:
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token

    if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
        adapter_kwargs["token"] = token

    if use_safetensors is None and not is_safetensors_available():
        use_safetensors = False
    if trust_remote_code is True:
        logger.warning(
            "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
            " ignored."
        )

    if commit_hash is None:
        if not isinstance(config, PretrainedConfig):
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file = cached_file(
                model_id,
                CONFIG_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            commit_hash = getattr(config, "_commit_hash", None)

    if is_peft_available():
        _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

        if _adapter_model_path is None:
            _adapter_model_path = find_adapter_config_file(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                _commit_hash=commit_hash,
                **adapter_kwargs,
            )
        if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
            with open(_adapter_model_path, "r", encoding="utf-8") as f:
                _adapter_model_path = model_id
                model_id = json.load(f)["base_model_name_or_path"]
    else:
        _adapter_model_path = None
    # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
    if load_in_4bit or load_in_8bit:
        if quantization_config is not None:
            raise ValueError(
                "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
                "`quantization_config` argument at the same time."
            )

        # preparing BitsAndBytesConfig from kwargs
        config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
        config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
        quantization_config, kwargs = BitsAndBytesConfig.from_dict(
            config_dict=config_dict, return_unused_kwargs=True, **kwargs
        )
        logger.warning(
            "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
            "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
        )

    user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else model_id
        config, _ = PretrainedConfig.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        # In case one passes a config to `from_pretrained` + "attn_implementation"
        # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
        # Please see: https://github.com/huggingface/transformers/issues/28038

        # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
        # we pop attn_implementation from the kwargs but this handles the case where users
        # passes manually the config to `from_pretrained`.
        config = copy.deepcopy(config)

        kwarg_attn_imp = kwargs.pop("attn_implementation", None)
        if kwarg_attn_imp is not None and config._attn_implementation != kwarg_attn_imp:
            config._attn_implementation = kwarg_attn_imp

    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    # index of the files.
    is_sharded = False

    assert model_id is not None
    model_id = str(model_id)
    is_local = os.path.isdir(model_id)
    assert not is_local

    # set correct filename
    if from_tf:
        filename = TF2_WEIGHTS_NAME
    elif from_flax:
        filename = FLAX_WEIGHTS_NAME
    elif use_safetensors is not False:
        filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
    else:
        filename = _add_variant(WEIGHTS_NAME, variant)

    try:
        # Load from URL or cache if already cached
        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "token": token,
            "user_agent": user_agent,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_gated_repo": False,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }
        resolved_archive_file = cached_file(model_id, filename, **cached_file_kwargs)

        # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
        # result when internet is up, the repo and revision exist, but the file does not.
        if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            resolved_archive_file = cached_file(
                model_id,
                _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                **cached_file_kwargs,
            )
            if resolved_archive_file is not None:
                is_sharded = True
            elif use_safetensors:
                if revision == "main":
                    resolved_archive_file, revision, is_sharded = auto_conversion(
                        model_id, **cached_file_kwargs
                    )
                cached_file_kwargs["revision"] = revision
                if resolved_archive_file is None:
                    raise EnvironmentError(
                        f"{model_id} does not appear to have a file named"
                        f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                        "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                        "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                    )
            else:
                # This repo has no safetensors file of any kind, we switch to PyTorch.
                filename = _add_variant(WEIGHTS_NAME, variant)
                resolved_archive_file = cached_file(
                    model_id, filename, **cached_file_kwargs
                )
        if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            resolved_archive_file = cached_file(
                model_id,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
                **cached_file_kwargs,
            )
            if resolved_archive_file is not None:
                is_sharded = True
        if resolved_archive_file is None:
            # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
            # message.
            has_file_kwargs = {
                "revision": revision,
                "proxies": proxies,
                "token": token,
            }
            if has_file(model_id, TF2_WEIGHTS_NAME, **has_file_kwargs):
                raise EnvironmentError(
                    f"{model_id} does not appear to have a file named"
                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                    " Use `from_tf=True` to load this model from those weights."
                )
            elif has_file(model_id, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                raise EnvironmentError(
                    f"{model_id} does not appear to have a file named"
                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                    " `from_flax=True` to load this model from those weights."
                )
            elif variant is not None and has_file(
                model_id, WEIGHTS_NAME, **has_file_kwargs
            ):
                raise EnvironmentError(
                    f"{model_id} does not appear to have a file named"
                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                    f" {variant}. Use `variant=None` to load this model from those weights."
                )
            else:
                raise EnvironmentError(
                    f"{model_id} does not appear to have a file named"
                    f" {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                    f" {FLAX_WEIGHTS_NAME}."
                )
    except EnvironmentError:
        # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
        # to the original exception.
        raise
    except Exception as e:
        # For any other exception, we throw a generic error.
        raise EnvironmentError(
            f"Can't load the model for '{model_id}'. If you were trying to load it"
            " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
            f" same name. Otherwise, make sure '{model_id}' is the correct path to a"
            f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
            f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
        ) from e

    # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    if is_sharded:
        # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, _ = get_checkpoint_shard_files(
            model_id,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )
    return


def fs_tf_tfs_cache_model(
    model_id,
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: bool = None,
    **kwargs
):
    from_pt = kwargs.pop("from_pt", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    _ = kwargs.pop("mirror", None)
    load_weight_prefix = kwargs.pop("load_weight_prefix", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    tf_to_pt_weight_rename = kwargs.pop("tf_to_pt_weight_rename", None)

    # Not relevant for TF models
    _ = kwargs.pop("adapter_kwargs", None)

    if use_auth_token is not None:
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token

    if trust_remote_code is True:
        logger.warning(
            "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
            " ignored."
        )

    user_agent = {"file_type": "model", "framework": "tensorflow", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if use_safetensors is None and not is_safetensors_available():
        use_safetensors = False

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else model_id
        config, model_kwargs = PretrainedConfig.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            _commit_hash=commit_hash,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    if commit_hash is None:
        commit_hash = getattr(config, "_commit_hash", None)

    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    # index of the files.
    is_sharded = False
    # Load model
    assert model_id is not None
    model_id = str(model_id)
    is_local = os.path.isdir(model_id)
    assert not is_local

    # set correct filename
    if from_pt:
        filename = WEIGHTS_NAME
    elif use_safetensors is not False:
        filename = SAFE_WEIGHTS_NAME
    else:
        filename = TF2_WEIGHTS_NAME

    try:
        # Load from URL or cache if already cached
        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "token": token,
            "user_agent": user_agent,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_gated_repo": False,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }
        resolved_archive_file = cached_file(model_id, filename, **cached_file_kwargs)

        # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
        # result when internet is up, the repo and revision exist, but the file does not.
        if resolved_archive_file is None and filename == SAFE_WEIGHTS_NAME:
            # Did not find the safetensors file, let's fallback to TF.
            # No support for sharded safetensors yet, so we'll raise an error if that's all we find.
            filename = TF2_WEIGHTS_NAME
            resolved_archive_file = cached_file(
                model_id, TF2_WEIGHTS_NAME, **cached_file_kwargs
            )
        if resolved_archive_file is None and filename == TF2_WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            resolved_archive_file = cached_file(
                model_id, TF2_WEIGHTS_INDEX_NAME, **cached_file_kwargs
            )
            if resolved_archive_file is not None:
                is_sharded = True
        if resolved_archive_file is None and filename == WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            resolved_archive_file = cached_file(
                model_id, WEIGHTS_INDEX_NAME, **cached_file_kwargs
            )
            if resolved_archive_file is not None:
                is_sharded = True
        if resolved_archive_file is None:
            # Otherwise, maybe there is a PyTorch or Flax model file.  We try those to give a helpful error
            # message.
            has_file_kwargs = {
                "revision": revision,
                "proxies": proxies,
                "token": token,
            }
            if has_file(model_id, SAFE_WEIGHTS_INDEX_NAME, **has_file_kwargs):
                is_sharded = True
                raise NotImplementedError(
                    "Support for sharded checkpoints using safetensors is coming soon!"
                )
            elif has_file(model_id, WEIGHTS_NAME, **has_file_kwargs):
                raise EnvironmentError(
                    f"{model_id} does not appear to have a file named"
                    f" {TF2_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                    " load this model from those weights."
                )
            else:
                raise EnvironmentError(
                    f"{model_id} does not appear to have a file named {WEIGHTS_NAME},"
                    f" {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
                )

    except EnvironmentError:
        # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
        # to the original exception.
        raise
    except Exception:
        # For any other exception, we throw a generic error.

        raise EnvironmentError(
            f"Can't load the model for '{model_id}'. If you were trying to load it"
            " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
            f" same name. Otherwise, make sure '{model_id}' is the correct path to a"
            f" directory containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
        )

    # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, _ = get_checkpoint_shard_files(
            model_id,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            _commit_hash=commit_hash,
        )
    return


def get_model_class_for_task(
    task: str,
    framework: str = "pt",
    model_type: Optional[str] = None,
    model_class_name: Optional[str] = None,
    library: str = "transformers",
) -> Tuple[List[str], Type]:
    # Modified from sources: TasksManager.get_model_class_from_task
    """
    Attempts to retrieve an AutoModel class from a task name.

    Args:
        task (`str`):
            The task required.
        framework (`str`, defaults to `"pt"`):
            The framework to use for the export.
        model_type (`Optional[str]`, defaults to `None`):
            The model type to retrieve the model class for. Some architectures need a custom class to be loaded,
            and can not be loaded from auto class.
        model_class_name (`Optional[str]`, defaults to `None`):
            A model class name, allowing to override the default class that would be detected for the task. This
            parameter is useful for example for "automatic-speech-recognition", that may map to
            AutoModelForSpeechSeq2Seq or to AutoModelForCTC.
        library (`str`, defaults to `transformers`):
                The library name of the model.

    Returns:
        The AutoModel class corresponding to the task.
    """
    task = task.replace("-with-past", "")
    task = TasksManager.map_from_synonym(task)

    TasksManager._validate_framework_choice(framework)

    if (framework, model_type, task) in TasksManager._CUSTOM_CLASSES:
        library, class_name = TasksManager._CUSTOM_CLASSES[(framework, model_type, task)]
        loaded_library = importlib.import_module(library)

        return getattr(loaded_library, class_name)
    else:
        if framework == "pt":
            tasks_to_model_loader = TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[library]
        else:
            tasks_to_model_loader = TasksManager._LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP[library]

        loaded_library = importlib.import_module(library)

        model_class_names = list()
        if model_class_name is None:
            if task not in tasks_to_model_loader:
                raise KeyError(
                    f"Unknown task: {task}. Possible values are: "
                    + ", ".join([f"`{key}` for {tasks_to_model_loader[key]}" for key in tasks_to_model_loader])
                )

            if isinstance(tasks_to_model_loader[task], str):
                model_class_name = tasks_to_model_loader[task]
                model_class_names.append(model_class_name)
            else:
                # automatic-speech-recognition case, which may map to several auto class
                if library == "transformers":
                    if model_type is None:
                        logger.warning(
                            f"No model type passed for the task {task}, that may be mapped to several loading"
                            f" classes ({tasks_to_model_loader[task]}). Defaulting to {tasks_to_model_loader[task][0]}"
                            " to load the model."
                        )
                        model_class_names.extend(list(tasks_to_model_loader[task]))
                    else:
                        for autoclass_name in tasks_to_model_loader[task]:
                            module = getattr(loaded_library, autoclass_name)
                            # TODO: we must really get rid of this - and _ mess
                            if (
                                model_type in module._model_mapping._model_mapping
                                or model_type.replace("-", "_") in module._model_mapping._model_mapping
                            ):
                                model_class_names.append(autoclass_name)

                        if len(model_class_names) == 0:
                            raise ValueError(
                                f"Unrecognized configuration classes {tasks_to_model_loader[task]} do not match"
                                f" with the model type {model_type} and task {task}."
                            )
                else:
                    raise NotImplementedError(
                        "For library other than transformers, the _TASKS_TO_MODEL_LOADER mapping should be one to one."
                    )
        else:
            model_class_names.append(model_class_name)

        return model_class_names, loaded_library


def cache_model_from_task(
    task,
    model_id,
    subfolder,
    revision,
    cache_dir,
    framework,
    torch_dtype,
    library_name,
    **model_kwargs,
):
    # Sources from TasksManager.get_model_from_task

    framework = TasksManager.determine_framework(model_id, subfolder=subfolder, framework=framework)

    original_task = task
    if task == "auto":
        task = TasksManager.infer_task_from_model(model_id, subfolder=subfolder, revision=revision)

    library_name = TasksManager.infer_library_from_model(
        model_id, subfolder, revision, cache_dir, library_name
    )

    model_type = None
    model_class_name = None
    kwargs = {"subfolder": subfolder, "revision": revision, "cache_dir": cache_dir, **model_kwargs}

    if library_name == "transformers":
        config = AutoConfig.from_pretrained(model_id, **kwargs)
        model_type = config.model_type.replace("_", "-")
        # TODO: if automatic-speech-recognition is passed as task, it may map to several
        # different auto class (AutoModelForSpeechSeq2Seq or AutoModelForCTC),
        # depending on the model type
        # if original_task in ["auto", "automatic-speech-recognition"]:
        if original_task == "automatic-speech-recognition" or task == "automatic-speech-recognition":
            if original_task == "auto" and config.architectures is not None:
                model_class_name = config.architectures[0]
        #^^^^^^^^^^^^^^^^^^^^ Serveral Different Model Classes ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    model_class_names, loaded_library = get_model_class_for_task(
        task, framework, model_type=model_type, model_class_name=model_class_name, library=library_name
    )

    if library_name == "timm":
        assert len(model_class_names) == 1
        assert model_class_names[0] == 'create_model'
        timm_cache_model(f"hf_hub:{model_id}", pretrained=True, exportable=True)
    elif library_name == "sentence_transformers":
        assert len(model_class_names) == 1
        assert model_class_names[0] == 'SentenceTransformer'
        cache_folder = cache_dir
        use_auth_token = model_kwargs.pop("use_auth_token", None)
        stfs_cache_model(model_id, cache_folder=cache_folder, use_auth_token=use_auth_token)
    else:
        try:
            if framework == "pt":
                kwargs["torch_dtype"] = torch_dtype
            fs_cache_model(library_name, model_class_names, model_id, **kwargs)
        except OSError:
            if framework == "pt":
                logger.info("Loading TensorFlow model in PyTorch before exporting.")
                kwargs["from_tf"] = True
                fs_cache_model(library_name, model_class_names, model_id, **kwargs)
            else:
                logger.info("Loading PyTorch model in TensorFlow before exporting.")
                kwargs["from_pt"] = True
                fs_cache_model(library_name, model_class_names, model_id, **kwargs)


def _infer_task_from_model_name_or_path(
    model_name_or_path: str, subfolder: str = "", revision: Optional[str] = None
) -> str:
    inferred_task_name = None
    is_local = os.path.isdir(os.path.join(model_name_or_path, subfolder))

    if is_local:
        # TODO: maybe implement that.
        raise RuntimeError("Cannot infer the task from a local directory yet, please specify the task manually.")
    else:
        if subfolder != "":
            raise RuntimeError(
                "Cannot infer the task from a model repo with a subfolder yet, please specify the task manually."
            )
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
        if getattr(model_info, "library_name", None) == "diffusers":
            class_name = model_info.config["diffusers"]["class_name"]
            inferred_task_name = "stable-diffusion-xl" if "StableDiffusionXL" in class_name else "stable-diffusion"
        elif getattr(model_info, "library_name", None) == "timm":
            inferred_task_name = "image-classification"
        else:
            pipeline_tag = getattr(model_info, "pipeline_tag", None)
            # The Hub task "conversational" is not a supported task per se, just an alias that may map to
            # text-generaton or text2text-generation.
            # The Hub task "object-detection" is not a supported task per se, as in Transformers this may map to either
            # zero-shot-object-detection or object-detection.
            if pipeline_tag is not None and pipeline_tag not in ["conversational", "object-detection"]:
                inferred_task_name = TasksManager.map_from_synonym(model_info.pipeline_tag)
            else:
                transformers_info = model_info.transformersInfo
                if transformers_info is not None and transformers_info.get("pipeline_tag") is not None:
                    inferred_task_name = TasksManager.map_from_synonym(transformers_info["pipeline_tag"])
                elif model_info.library_name == "sentence-transformers":
                    inferred_task_name = "sentence-similarity"
                else:
                    # transformersInfo does not always have a pipeline_tag attribute
                    class_name_prefix = ""
                    if is_torch_available():
                        tasks_to_automodels = TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[
                            model_info.library_name
                        ]
                    else:
                        tasks_to_automodels = TasksManager._LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP[
                            model_info.library_name
                        ]
                        class_name_prefix = "TF"

                    auto_model_class_name = transformers_info["auto_model"]
                    if not auto_model_class_name.startswith("TF"):
                        auto_model_class_name = f"{class_name_prefix}{auto_model_class_name}"
                    for task_name, class_name_for_task in tasks_to_automodels.items():
                        if class_name_for_task == auto_model_class_name:
                            inferred_task_name = task_name
                            break

    if inferred_task_name is None:
        raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")
    return inferred_task_name


def infer_task_from_model(
    model: Union[str, "PreTrainedModel", "TFPreTrainedModel", Type],
    subfolder: str = "",
    revision: Optional[str] = None,
) -> str:
    """
    Infers the task from the model repo.

    Args:
        model (`str`):
            The model to infer the task from. This can either be the name of a repo on the HuggingFace Hub, an
            instance of a model, or a model class.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the model files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        revision (`Optional[str]`,  defaults to `None`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
    Returns:
        `str`: The task name automatically detected from the model repo.
    """
    is_torch_pretrained_model = is_torch_available() and isinstance(model, PreTrainedModel)
    is_tf_pretrained_model = is_tf_available() and isinstance(model, TFPreTrainedModel)
    task = None
    if isinstance(model, str):
        task = _infer_task_from_model_name_or_path(model, subfolder=subfolder, revision=revision)
    elif is_torch_pretrained_model or is_tf_pretrained_model:
        task = TasksManager._infer_task_from_model_or_model_class(model=model)
    elif inspect.isclass(model):
        task = TasksManager._infer_task_from_model_or_model_class(model_class=model)

    if task is None:
        raise ValueError(f"Could not infer the task from {model}.")

    return task


def cache_model(model_id, cache_dir, monolith) -> Dict:
    subfolder = ""
    revision = None
    use_auth_token = None
    local_files_only = False
    force_download = False
    trust_remote_code = False
    device = 'cpu'
    framework = None


    dtype = "fp32"
    task = "auto"
    original_task = task
    task = TasksManager.map_from_synonym(task)
    framework = TasksManager.determine_framework(model_id, subfolder=subfolder, framework=framework)
    library_name = TasksManager.infer_library_from_model(model_id)
    torch_dtype = None
    if framework == "pt":
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
    if task == "auto":
        try:
            task = infer_task_from_model(model_id)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    custom_architecture = False
    loading_kwargs = {}
    if library_name == "transformers":
        config = AutoConfig.from_pretrained(
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        model_type = config.model_type.replace("_", "-")

        if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
            custom_architecture = True
        elif task not in TasksManager.get_supported_tasks_for_model_type(
            model_type, "onnx", library_name=library_name
        ):
            if original_task == "auto":
                autodetected_message = " (auto-detected)"
            else:
                autodetected_message = ""
            model_tasks = TasksManager.get_supported_tasks_for_model_type(
                model_type, exporter="onnx", library_name=library_name
            )
            raise ValueError(
                f"Asked to export a {model_type} model for the task {task}{autodetected_message}, but the Optimum ONNX exporter only supports the tasks {', '.join(model_tasks.keys())} for {model_type}. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task {task} to be supported in the ONNX export for {model_type}."
            )

        # TODO: Fix in Transformers so that SdpaAttention class can be exported to ONNX. `attn_implementation` is introduced in Transformers 4.36.
        if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED and _transformers_version >= version.parse("4.35.99"):
            loading_kwargs["attn_implementation"] = "eager"

    cache_model_from_task(
        task,
        model_id,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework=framework,
        torch_dtype=torch_dtype,
        device=device,
        library_name=library_name,
        **loading_kwargs,
    )

    cache_args_dict = dict(
        task=task,
        model_id=model_id,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        framework=framework,
        torch_dtype=torch_dtype,
        device=device,
        library_name=library_name,
        loading_kwargs=loading_kwargs,
        opset=None,
        optimize=None,
        monolith=monolith,
        no_post_process=False,
        atol=None,
        do_validation=True,
        model_kwargs=None,
        custom_onnx_configs=None,
        fn_get_submodels=None,
        use_subprocess=False,
        _variant="default",
        legacy=False,
        trust_remote_code=trust_remote_code,
        no_dynamic_axes=False,
        do_constant_folding=True,
        custom_architecture=custom_architecture,
        original_task=original_task
    )

    return cache_args_dict
