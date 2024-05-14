# Transformers.js Convert To Younger.Instance

## 安装

0. 安装`transformers.js` (`transformers`版本不宜太高，`onnx`版本不宜太低，`requirements.txt`折衷了`transformers.js`官方给出的依赖和`Younger`所需依赖。)
``` bash
pip install -r requirements.txt
```

## 运行

0. **关于`convert/main.py`的功能介绍:**
该模块可以将输入的一组由若干`model_id`标识的`HuggingFace`模型转换成`Younger.Instance`。
特别的，`--model_id`为`convert/supported_models.py`文件中指示的`Transformers.js`所支持的`HuggingFace`中不同`task`下的模型架构。
在官方仓库中查看[详细信息](https://github.com/xenova/transformers.js?tab=readme-ov-file#supported-tasksmodels)。

0. 任意位置新建一个执行脚本，如下: [**注意** - 应额外指定其中的`shell`变量，本说明文档不做详细展示]
``` bash
current_dir=$(pwd)
MODEL_IDS_FILEPATH=$(realpath ${MODEL_IDS_FILEPATH})
CACHE_DIRPATH=$(realpath ${CACHE_DIRPATH})
SAVE_DIRPATH=$(realpath ${SAVE_DIRPATH})
trap 'popd' EXIT
pushd ${HOME}/${CLONE_PATH}/Younger/younger/datasets/constructors/huggingface/scripts/transformers.js
bash ./convert.sh $MODEL_IDS_FILEPATH} ${CACHE_DIRPATH} ${SAVE_DIRPATH}
```
`MODEL_IDS_FILEPATH`应为`json`格式的文件，包含一个`list`，如下格式：
``` json
[
  "albert-base-v2",
  "microsoft/beit-base-patch16-224"
]
```

`CACHE_DIRPATH`应为`convert/main.py`转出的`onnx`模型的保存位置。

`SAVE_DIRPATH`应为`convert/main.py`最终输出`Younger.Instances`的目录

## 代码
该脚本除以下所列文件外，基于`transformers.js`官方代码进行了修改：
1. `requirements.txt`
2. `README.md`
3. `README.ZH.md`
4. `convert.sh`

特别的，`convert/main.py`:
- 添加了参数`remove_other_files`，作用为转化`instance`后删除途中下载的`onnx`模型以及相关信息文件
- 添加了注释为step3、5的代码段，作用为将`onnx`转换为`instance`，并执行上一步中的`remove_other_files`
- 添加了`onnx2instance`函数