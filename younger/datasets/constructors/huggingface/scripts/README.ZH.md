1. **依赖**
   
   pip install -r requirements.txt（YoungBench使用的ExportONNX无法运行convert代码，transformers版本不宜太高，onnx版本不宜太低，这个requirements折中了transformers.js官方给出的依赖和Younger所需依赖）

2. **运行**
   
   执行 convert2onnx2insance.sh 
   
   其中：python path/to/scripts/convert.py可以将输入的一组task、model_id标识的模型转换成instance

       特别地，--task 与--model_id 为 scripts/supported_models.py 文件中指示的Transformers.js支持的huggingface中不同task下的模型。如下图所示：

<img src="file:///Users/zrsion/Downloads/1715435118596.jpg" title="" alt="" width="509">

也可以在(https://github.com/xenova/transformers.js?tab=readme-ov-file#supported-tasksmodels) 获取详细信息

shell代码中JSON_PATH为我测试转换功能时使用的task：model_id字典，

如下格式：{

        "fill-mask": "albert-base-v2",

        "image-classification": "microsoft/beit-base-patch16-224"

}

OUTPUT_PARENT_DIR为最终输出instances的目录

3. **代码**：
   
   整个scripts为拉取官方代码结果，我做了以下修改：
   
   convert.py：
   
   a.添加了参数remove_other_files，作用为转化instance后删除途中下载的onnx模型以及相关信息文件
   
   b.代码中添加了注释为step3、5的代码段，作用为将onnx转换为instance，并执行a中的remove_other_files
   
   c.添加了onnx2instance函数


