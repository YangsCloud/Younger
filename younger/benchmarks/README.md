## Table of Contents
<!-- vscode-markdown-toc -->
* 1. [Usage](#Usage)
	* 1.1. [Prepare](#Prepare)
		* 1.1.1. [Download Younger](#DownloadYounger)
		* 1.1.2. [Get Operator/Network Embedding](#GetOperatorNetworkEmbedding)
	* 1.2. [Benchmark Generation](#BenchmarkGeneration)
	* 1.3. [Analysis and Visualization](#AnalysisandVisualization)
		* 1.3.1. [Prepare](#Prepare-1)
		* 1.3.2. [Statistical Analysis](#StatisticalAnalysis)
		* 1.3.3. [Structral Analysis](#StructralAnalysis)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

##  1. <a name='Usage'></a>Usage

###  1.1. <a name='Prepare'></a>Prepare

Overall Directory Structure:
```
YoungBench/
├── Embedding/
├── Analysis/
│   └── ...
└── Assets/
    ├── younger/
    │   ├── detailed_filter_series_without_attributes_paper/
    │   │   ├── 00042d1e33bfd0b910be59c7dff8a364/
    │   │   ├── 00044cab84b1a002bb9f03750d9a74b1/
    │   │   ├── ... (more instances)
    │   │   └── fffbe3b36222ebc90235ae5f5443845b/
    │   ├── detailed_filter_series_without_attributes_paper.tgz
    │   └── version
    └── competitors/
        ├── mlperf_v0.5/
        ├── mlperf_v4.1/
        └── phoronix/
```

####  1.1.1. <a name='DownloadYounger'></a>Download Younger
Create a directory `YoungBench/Assets/younger`:
```shell
mkdir -p ./YoungBench/Assets/younger
```

Create a version file `version` which contains a line `paper`:
```shell
echo "paper" > ./YoungBench/Assets/younger/version
```

Run the script:
```shell
younger benchmarks prepare --bench-dirpath ./YoungBench/Assets/younger/ --dataset-dirpath ./YoungBench/Assets/younger/ --version younger
```

You'll get a sub-directory under `younger`:
```
younger/
├── detailed_filter_series_without_attributes_paper/
│   ├── 00042d1e33bfd0b910be59c7dff8a364/
│   ├── 00044cab84b1a002bb9f03750d9a74b1/
│   ├── ... (more instances)
│   └── fffbe3b36222ebc90235ae5f5443845b/
├── detailed_filter_series_without_attributes_paper.tgz
└── version
```

####  1.1.2. <a name='GetOperatorNetworkEmbedding'></a>Get Operator/Network Embedding

##### Download Pre-Trained Embedding
**`TBD`**

##### From Scratch
This project utilizes operator embeddings from the Operator Type Classification task to obtain and calculate operator and network embeddings.

**NOTE:** **Q**: What is the Operator Type Classification task?
**A**: The model needs to predict the class of the masked node with the information from other nodes and their connections within a subgraph or the entire graph. This project primarily uses subgraphs for training making the training progress efficient and effective.

1. Create the directory `YoungBench/Embedding`:
```shell
mkdir -p ./YoungBench/Embedding
```

2. Change into the directory `YoungBench/Embedding`:
```shell
cd YoungBench/Embedding
```

3. Extract subgraphs (only contains official ONNX operators) from Younger
```shell
DATASET_DIRPATH="../Assets/younger/detailed_filter_series_without_attributes_paper/"
younger datasets split --mode random --version subgraphs \
    --dataset-dirpath ${DATASET_DIRPATH}/ \
    --save-dirpath . \
    --subgraph-sizes 5 6 7 8 9 10 11 12 13 14 15 \
    --subgraph-number 500 \
    --retrieve-try 3600 \
    --node-size-lbound 4 \
    --train-proportion 98.0 \
    --valid-proportion 1.0 \
    --test-proportion 1.0 \
    --seed 666666 \
    --logging-filepath extract_subgraphs.log
```
or one can extract subgraphs also contains `'com.microsoft'` operators with argument `--allow-domain 'com.microsoft'`, as follows:
```shell
DATASET_DIRPATH="../Assets/younger/detailed_filter_series_without_attributes_paper/"
younger datasets split --mode random --version subgraphs_with_ms \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath . \
    --subgraph-sizes 5 6 7 8 9 10 11 12 13 14 15 \
    --subgraph-number 500 \
    --retrieve-try 3600 \
    --node-size-lbound 4 \
    --train-proportion 98.0 \
    --valid-proportion 1.0 \
    --test-proportion 1.0 \
    --seed 666666 \
    --logging-filepath extract_subgraphs_with_ms.log \
    --allow-domain 'com.microsoft'
```
or just download the preprocessed dataset with extracted subgraphs:

**`TBD`**

4. Create the deep learning configuration file `model.toml`:
```toml
# model.toml

mode = 'Train'

[dataset]
train_dataset_dirpath = "subgraphs/train"
valid_dataset_dirpath = "subgraphs/valid"
test_dataset_dirpath  = "subgraphs/test"

# train_dataset_dirpath = "subgraphs_with_ms/train"
# valid_dataset_dirpath = "subgraphs_with_ms/valid"
# test_dataset_dirpath  = "subgraphs_with_ms/test"

dataset_name = 'YoungBench_Embedding'
encode_type = 'operator'
standard_onnx = true
worker_number = 32

[model]
model_type = 'SAGE_NP'
node_dim = 512
hidden_dim = 256
dropout = 0.5

[optimizer]
learning_rate = 1e-3
weight_decay = 5e-5

[scheduler]
# step_size=40000
# gamma=0.5

[embedding]
name = "GCN_Subgraphs"
activate = true
embedding_dirpath = "."

[api]
meta_filepath = ""
onnx_model_dirpath = ""

[logging]
name = "Subgraphs"
# name = "Subgraphs_With_MS"
mode = "both"
filepath = "./subgraphs.log"
# filepath = "./subgraphs_with_ms.log"
```
The argument `model_type` can be set to either `'GCN_NP'` or `SAGE_NP`, and the dataset directory can be set to either `subgraphs` or `subgraphs_with_ms` to obtain different model performance or to servce various experimental purposes.

5. Change the name the sub-directory:
```shell
mv subgraphs/train/item subgraphs/train/YoungBench_Embedding_Raw
mv subgraphs/valid/item subgraphs/valid/YoungBench_Embedding_Raw
mv subgraphs/test/item  subgraphs/test/YoungBench_Embedding_Raw
```

6. Create the training script `train.sh`:
```shell
#!/bin/bash
# train.sh
THIS_NAME=GCN

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_DIRPATH=./Checkpoint/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning train \
  --task-name node_prediciton \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-dirpath ${CHECKPOINT_DIRPATH} \
  --checkpoint-name ${CHECKPOINT_NAME} \
  --keep-number 200 \
  --train-batch-size 512 --valid-batch-size 512 --shuffle \
  --life-cycle 500 --report-period 10 --update-period 1 \
  --train-period 100 --valid-period 100 \
  --device GPU \
  --world-size 4 --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} --master-rank ${MASTER_RANK} \
  --seed 12345
```

7. Start training:
```shell
chmod +x train.sh
./train.sh
```

8. Stop when the training/validation loss converges and test the corresponding checkpoint: (A). First, change the value of `mode` argument to `'Test'`; (B). Then, create shell script `test.sh` like `train.sh`, and run the script `test.sh`:
```shell
#!/bin/bash
# test.sh

THIS_NAME=GCN

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_FILEPATH=./Checkpoint/${THIS_NAME}/GCN_Epoch_100_Step_33800.cp
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning test \
  --task-name node_prediciton \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-filepath ${CHECKPOINT_FILEPATH} \
  --test-batch-size 512 \
```

```shell
chmod +x test.sh
./test.sh
```

9. Select an appropriate checkpoint to get the `'weights'` and `'op_dict'` of the operator embeddings, and change the value of `mode` argument to `'Test'` and `embedding.activate` to `true`, like:
```toml
mode = "Test"

# ...
# ^ Other Settings

[embedding]
name = "GCN_Subgraphs"
activate = true
embedding_dirpath = "."

# v Other Settings
# ...

```

10. Run shell script `test.sh` again. The file `YBEmb_*_weights.npy` and `YBEmb_*_op_dict.json` will be saved under the directory `YoungBench/Embedding`:
```shell
./test.sh
```
Finally the directory `YoungBench/Embedding` looks like:
```shell
Embedding/
├── Checkpoint/
├── extract_subgraphs.log
├── extract_subgraphs_with_ms.log
├── model.toml
├── YBEmb_GCN_Subgraphs.meta
├── YBEmb_GCN_Subgraphs_op_dict.json
├── subgraphs/
├── subgraphs_with_ms/
├── test.sh
├── train.sh
└── YBEmb_GCN_Subgraphs_weights.npy
```

###  1.2. <a name='BenchmarkGeneration'></a>Benchmark Generation

###  1.3. <a name='AnalysisandVisualization'></a>Analysis and Visualization

####  1.3.1. <a name='Prepare-1'></a>Prepare

Download Competitors under directory `YoungBench/Assets/competitors`:

* **Download MLPerf ONNXs or Instances:** Please refer to README.md on HuggingFace (YoungBench-Assets) [MLPerf-Auto-Download](https://huggingface.co/datasets/AIJasonYoung/YoungBench-Assets#automatically)

* **Download Phoronix ONNXs or Instances:** Please refer to README.md on HuggingFace (YoungBench-Assets) [Phoronix-Auto-Download](https://huggingface.co/datasets/AIJasonYoung/YoungBench-Assets#automatically)

Create a directory `YoungBench/Analysis`:
```shell
mkdir -p ./YoungBench/Analysis
```

Change the working directory to `YoungBench/Analysis`:
```shell
cd ./YoungBench/Analysis
```

####  1.3.2. <a name='StatisticalAnalysis'></a>Statistical Analysis

Create a dataset indices file `other_dataset_indices` which contains multiple lines `Dataset_Name: Dataset_Directory`, suppose that one want to analysis both MLPerf and Phoronix, the file can contain lines:
```shell
mlperf_v0.5: ../Assets/competitors/mlperf_v0.5/instances
mlperf_v4.1: ../Assets/competitors/mlperf_v4.1/instances
phoronix: ../Assets/competitors/phoronix/instances
```

Run command to perform analysis:
```shell
younger benchmarks analyze --younger-dataset-dirpath ../Assets/younger/detailed_filter_series_without_attributes_paper --statistics-dirpath . --other-dataset-indices-filepath ./other_dataset_indices
```

All analysis results will be placed at directory `YoungBench/Analysis`, and the structure is as below:
```
Analysis/
├── other_dataset_indices
├── statistics_compare_mlperf_v0.5.json
├── statistics_compare_mlperf_v4.1.json
├── statistics_compare_phoronix.json
├── statistics_mlperf_v0.5.json
├── statistics_mlperf_v0.5.xlsx
├── statistics_mlperf_v4.1.json
├── statistics_mlperf_v4.1.xlsx
├── statistics_phoronix.json
├── statistics_phoronix.xlsx
├── statistics_younger.json
└── statistics_younger.xlsx
```

####  1.3.3. <a name='StructralAnalysis'></a>Structral Analysis