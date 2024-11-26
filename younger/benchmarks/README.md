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
		* 1.3.4. [Analysis Results](#AnalysisResults)

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
younger datasets split --mode random --version YoungBench_Embedding \
    --dataset-dirpath ${DATASET_DIRPATH}/ \
    --save-dirpath . \
    --subgraph-sizes 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --subgraph-number 300 \
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
younger datasets split --mode random --version YoungBench_Embedding_With_MS \
    --dataset-dirpath ${DATASET_DIRPATH} \
    --save-dirpath . \
    --subgraph-sizes 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
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
# mode = 'Test'

[dataset]
train_dataset_dirpath = "YoungBench_Embedding/train"
valid_dataset_dirpath = "YoungBench_Embedding/valid"
test_dataset_dirpath =  "YoungBench_Embedding/test"

dataset_name = 'YoungBench_Embedding'
encode_type = 'operator'
standard_onnx = true
worker_number = 32
mask_ratio = 0.15

[model]
model_type = 'MAEGIN'
node_dim = 512
hidden_dim = 1024
dropout = 0.2
layer_number = 3

[optimizer]
lr = 0.001
eps = 1e-8
weight_decay = 0.01
amsgrad = true

[scheduler]
start_factor = 0.1
warmup_steps = 1500
total_steps = 150000
last_step = -1

[cli]
node_size_limit = 4
meta_filepath = "YoungBench_Embedding/test/meta.json"
result_filepath = "Results/Phoronix.pkl"
instances_dirpath = "../Assets/competitors/phoronix/instances"
# result_filepath = "Results/MLPerf_V4.1.pkl"
# instances_dirpath = "../Assets/competitors/mlperf_v4.1/instances"
# result_filepath = "Results/MLPerf_V0.5.pkl"
# instances_dirpath = "../Assets/competitors/mlperf_v0.5/instances"
# result_filepath = "Results/Younger.pkl"
# instances_dirpath = "../Assets/younger/detailed_filter_series_without_attributes_paper/"

[logging]
name = "YoungBench_Embedding_MAEGIN"
mode = "both"
filepath = "./youngbench_embedding_maegin.log"

# name = "YoungBench_Embedding_With_MS_MAEGIN"
# mode = "both"
# filepath = "./youngbench_embedding_with_ms_maegin.log"

```
The the arguments can be changed and the dataset directory can be set to either `YoungBench_Embedding` or `YoungBench_Embedding_With_MS` to obtain different model performance or to servce various experimental purposes.

5. Change the name of the sub-directory:
```shell
mv YoungBench_Embedding/train/item YoungBench_Embedding/train/YoungBench_Embedding_Raw
mv YoungBench_Embedding/valid/item YoungBench_Embedding/valid/YoungBench_Embedding_Raw
mv YoungBench_Embedding/test/item  YoungBench_Embedding/test/YoungBench_Embedding_Raw
```

6. Create the training script `train.sh`:
```shell
#!/bin/bash
# train.sh
THIS_NAME=MAEGIN

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_DIRPATH=./Checkpoint/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning train \
  --task-name ssl_prediction \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-dirpath ${CHECKPOINT_DIRPATH} \
  --checkpoint-name ${CHECKPOINT_NAME} \
  --keep-number 200 \
  --train-batch-size 512 --valid-batch-size 512 --shuffle \
  --life-cycle 200 --report-period 10 --update-period 1 \
  --train-period 200 --valid-period 200 \
  --device GPU \
  --world-size 1 --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} --master-rank ${MASTER_RANK} \
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

THIS_NAME=MAEGIN

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_FILEPATH=./Checkpoint/${THIS_NAME}/MAEGIN_Epoch_150_Step_185800.cp
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning test \
  --task-name ssl_prediction \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-filepath ${CHECKPOINT_FILEPATH} \
  --test-batch-size 512 \
```

```shell
chmod +x test.sh
./test.sh
```

9. Select an appropriate checkpoint, and change the value of `mode` argument to `'Test'` and run the shell script `test.sh`, like:
```toml
mode = "Test"

# v Other Settings
# ...

```

```shell
./test.sh
```

10. Select an appropriate checkpoint to get `'emb_dict's` of the operator embeddings and graph embeddings for each Benchmarks (e.g. MLPerf, Phoronix) or Datasets (e.g. Younger). (A). First, change the value of `mode` argument to `'CLI'`; (B). Then, create shell script `cli.sh` like `test.sh`, and run the script `cli.sh`, like:
```toml
mode = "CLI"

# v Other Settings
# ...

```

```shell
#!/bin/bash
# cli.sh

THIS_NAME=MAEGIN

CONFIG_FILEPATH=${THIS_NAME}.toml
CHECKPOINT_FILEPATH=./Checkpoint/${THIS_NAME}/MAEGIN_Epoch_150_Step_185800.cp
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning cli \
  --task-name ssl_prediction \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-filepath ${CHECKPOINT_FILEPATH}
```

```shell
chmod +x cli.sh
./cli.sh
```

```shell
./cli.sh
```

11. The file `Results/*.pkl` will be saved under the directory `YoungBench/Embedding`. Finally the directory `YoungBench/Embedding` looks like:
```shell
Embedding/
├── Checkpoint/
├── cli.sh
├── Results/
│   ├── Phoronix.pkl
│   ├── MLPerf_V0.5.pkl
│   ├── MLPerf_V4.1.pkl
│   └── Younger.pkl
├── extract_subgraphs.log
├── extract_subgraphs.sh
├── extract_subgraphs_with_ms.log
├── extract_subgraphs_with_ms.sh
├── model.toml
├── test.sh
├── train.sh
├── YoungBench_Embedding/
├── youngbench_embedding_maegin.log
├── YoungBench_Embedding_With_MS/
└── youngbench_embedding_with_ms_maegin.log
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

Create the configuration file `analysis.toml`:
```toml
[stc]
younger_emb = { name = "Younger", path = "../Embedding/Results/Younger.pkl" }
compare_embs = [
    { name = "MLPerf_V0.5", path = "../Embedding/Results/MLPerf_V0.5.pkl" },
    { name = "MLPerf_V4.1", path = "../Embedding/Results/MLPerf_V4.1.pkl" },
    { name = "Phoronix",    path = "../Embedding/Results/Phoronix.pkl" }
]

op_cluster_type = "HDBSCAN"
op_cluster_kwargs = { prediction_data = true }
dag_cluster_type = "HDBSCAN"
dag_cluster_kwargs = { prediction_data = true }

op_reducer_type = "UMAP"
op_reducer_kwargs = {}
dag_reducer_type = "UMAP"
dag_reducer_kwargs = {}

[sts]
younger_dataset = { name = "Younger", path = "../Assets/younger/detailed_filter_series_without_attributes_paper/" }
compare_datasets = [
    { name = "MLPerf_V0.5", path = "../Assets/competitors/mlperf_v0.5/instances" },
    { name = "MLPerf_V4.1", path = "../Assets/competitors/mlperf_v4.1/instances" },
    { name = "Phoronix",    path = "../Assets/competitors/phoronix/instances" }
]

```

####  1.3.2. <a name='StatisticalAnalysis'></a>Statistical Analysis
Run command to perform analysis:
```shell
younger benchmarks analyze --results-dirpath . --configuration-filepath configuration.toml --mode sts
```

####  1.3.3. <a name='StructralAnalysis'></a>Structral Analysis
Run command to perform analysis:
```shell
younger benchmarks analyze --results-dirpath . --configuration-filepath configuration.toml --mode stc
```

####  1.3.4. <a name='AnalysisResults'></a>Analysis Results
All analysis results will be placed at directory `YoungBench/Analysis`, and the structure is as below:
```
Analysis/
├── statistical/
│   ├── sts_results_compare_mlperf_v0.5.json
│   ├── sts_results_compare_mlperf_v4.1.json
│   ├── sts_results_compare_phoronix.json
│   ├── sts_results_mlperf_v0.5.json
│   ├── sts_results_mlperf_v0.5.xlsx
│   ├── sts_results_mlperf_v4.1.json
│   ├── sts_results_mlperf_v4.1.xlsx
│   ├── sts_results_phoronix.json
│   ├── sts_results_phoronix.xlsx
│   ├── sts_results_younger.json
│   └── sts_results_younger.xlsx
├── structural/
│   ├── stc_visualization_sketch_20241111_111111.pdf
│   ├── stc_visualization_sketch_compare_MLPerf_V0.5_20241111_111111.pdf
│   ├── stc_visualization_sketch_compare_MLPerf_V4.1_20241111_111111.pdf
│   └── stc_visualization_sketch_compare_Phoronix_20241111_111111.pdf
└── configuration.toml
```
