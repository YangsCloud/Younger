## Usage

### Prepare

Overall Directory Structure:
```
YoungBench/
├── Embedding/
│   ├── 
│   ├── 
│   ├── ... (more instances)
│   └── 
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

#### Download Younger
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

#### Get Operator/Network Embedding

##### Download Pre-Trained Embedding

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

4. Create the training configuration file `model.toml`:
```toml
```

5. Now, train the model!
```shell
```


### Benchmark Generation

### Analysis and Visualization

#### Prepare

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

#### Statistical Analysis

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
YoungBench/
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

#### Structral Analysis