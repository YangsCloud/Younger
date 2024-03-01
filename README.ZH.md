# YoungBench

[English Documentation](./README.md)

## 依赖

1. conda create -n ExportONNX python=3.9

2. pip install optimum[exporters] YoungToolkit diffusers timm transformers sentence-transformers accelerator

## 数据集构建

预训练模型 (PyTorch/TensorFlow) -> ONNX 模型 -> Network (Directed Acyclic Graph)

### 从模型数据库获取预训练模型

#### HuggingFace Model Hub

1. HuggingFace模型下载和转换相关的环境变量设置在 [constants.sh](./constants.sh) 中，只需要修改`${HF_PATH}`变量，下载脚本和转换工具的结果都会输出在`${HF_PATH}`目录下。

2. [get_hf_models_info.sh](./get_hf_models_info.sh) 脚本获取`5`类预训练模型的信息，并将获取的模型信息存在`${HF_PATH}`目录下对应的文件夹`${<?>_PATH}/model_infos`中，`5`类预训练模型及其对应的`<?>`值如下 ([参数说明](./youngbench/dataset/scripts/get_hf_models_info.py))：

    * `4`个模型库 (Optimum库可以将库内预训练模型转换为ONNX格式)
        * [Timm](https://huggingface.co/models?library=timm&sort=likes) `(<?> = TIMM)`
        * [Diffusers](https://huggingface.co/models?library=diffusers&sort=likes) `(<?> = DFS)`
        * [Transformers](https://huggingface.co/models?library=transformers&sort=likes) `(<?> = TFS)`
        * [Sentence-Transformers](https://huggingface.co/models?library=sentence-transformers&sort=likes) `(<?> = STFS)`
    * 包含[ONNX标签](https://huggingface.co/models?library=onnx&sort=likes)的预训练模型 `(<?> = ONNX)`

3. [get_hf_models.sh](./get_hf_models.sh) 脚本根据上面获取的模型信息下载模型，已经下载的模型会被记录在`${HF_CACHE_FLAG_PATH}`标记文件中，模型将被下载至`${HF_CACHE_PATH}`缓存目录 ([参数说明](./youngbench/dataset/scripts/get_hf_models.py))。
