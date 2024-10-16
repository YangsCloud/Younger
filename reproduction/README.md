
---

# Reproducing the Experiments

To reproduce the experiments of this paper, follow the instructions below.

## Dependencies

Ensure you have Python 3.10 installed (versions < 3.10 are not allowed). Install the necessary packages using the following command:

```sh
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Additionally, install the following Python packages:

- `torch_geometric`
- `onnx`
- `matplotlib`

Finally, set up the project:

```sh
python setup.py build develop
```

## How to Run

You can create a `.sh` file and a `.toml` file to run the experiments. 

Assume that your directory structure is as follows:

```
LocalDataflow/
├── checkpoints  
├── configurations  
├── LocalDataflowDataset  
├── Logs  
└── scripts
```
### Configuration File (`gat_lp_operator.toml`)
Next, run the following command:
```
cd configurations
```
Taking local data flow based on GAT as an example. Create a `.toml` file with the following content:

```toml
mode = 'Train'

[model]
model_type = 'GAT_LP'
node_dim = 1024
hidden_dim = 512
output_dim = 256

[optimizer]
learning_rate = 1e-3
weight_decay = 5e-5

[dataset]
train_dataset_dirpath = "../LocalDataflowDataset/train"
valid_dataset_dirpath = "../LocalDataflowDataset/valid"
test_dataset_dirpath = "../LocalDataflowDataset/test"
encode_type = 'operator'

worker_number = 8

[logging]
name = "GAT_LP_operator"
mode = "both"
filepath = "../Logs/gat_lp_operator.log"
```

### Shell Script (`run_experiment.sh`)

Run the following command:
```
cd ../scripts
```
Create a `.sh` file with the following content:

```sh
#!/bin/bash

THIS_NAME=gat_lp_operator

CONFIG_FILEPATH=../configurations/${THIS_NAME}.toml
CHECKPOINT_DIRPATH=../checkpoints/checkpoint-gat-lp/${THIS_NAME}
CHECKPOINT_NAME=${THIS_NAME}
MASTER_ADDR=localhost
MASTER_PORT=16161
MASTER_RANK=0
LOGS_DIRPATH=../Logs
DATASET_DIRPATH=../LocalDataflowDataset
TRAIN_DIRPATH=${DATASET_DIRPATH}/train
VALID_DIRPATH=${DATASET_DIRPATH}/valid
TEST_DIRPATH=${DATASET_DIRPATH}/test

mkdir -p ${CHECKPOINT_DIRPATH}
mkdir -p ${LOGS_DIRPATH}
mkdir -p ${TRAIN_DIRPATH}
mkdir -p ${VALID_DIRPATH}
mkdir -p ${TEST_DIRPATH}

CUBLAS_WORKSPACE_CONFIG=:4096:8 younger applications deep_learning train \
  --task-name link_prediction \
  --config-filepath ${CONFIG_FILEPATH} \
  --checkpoint-dirpath ${CHECKPOINT_DIRPATH} --checkpoint-name ${CHECKPOINT_NAME} --keep-number 200 \
  --train-batch-size 1 --valid-batch-size 1 --shuffle \
  --life-cycle 6 --report-period 50 --update-period 1 --train-period 200 --valid-period 200 \
  --device GPU \
  --world-size 4 --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} --master-rank ${MASTER_RANK} \
  --seed 12345
```

**Note:** In the context of this script, 'operator' refers to 'w/o Attributes' in the paper, while 'node' means 'w/ Attributes'.


## Running the Experiment

Finally, run the `.sh` file:

```sh
./run_experiment.sh
```
The experiments of local operator can be reproduced in the same way.

---


