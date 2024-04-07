from huggingface_hub import list_metrics
from huggingface_hub.repocard_data import model_index_to_eval_results
import pathlib
import json
from huggingface_hub import utils, HfFileSystem, ModelCard, ModelCardData, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from younger.datasets.constructors.huggingface.utils import extract_candidate_metrics_from_readme

readme = """
---
license: apache-2.0
tags:
- text2text-generation
- generated_from_trainer
metrics:
- rouge
- bleu
datasets:
- domenicrosati/QA2D
model-index:
- name: QA2D-t5-base
  results:
  - task:
      name: Question to Declarative Sentence
      type: text2text-generation
    dataset:
      name: domenicrosati/QA2D
      type: domenicrosati/QA2D
      args: plain_text
    metrics:
    - name: Rouge1
      type: rouge
      value: 90.1064
    - name: Rouge2
      type: rouge
      value: 82.378
    - name: Rougel
      type: rouge
      value: 85.7963
    - name: Rougelsum
      type: rouge
      value: 85.8004
    - name: Bleu
      type: bleu
      value: 72.7328
widget:
- text: "where in the world is carmen sandiego. she is in abruzzo"
  example_title: "Where is Carmen Sandiego?"
- text: "which province is halifax in. nova scotia"
  example_title: "A Halifact"
---
# QA2D-t5-base

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on [QA2D](https://huggingface.co/datasets/domenicrosati/QA2D).
It achieves the following results on the evaluation set:
- Loss: 0.2563
- Rouge1: 90.1064
- Rouge2: 82.378
- Rougel: 85.7963
- Rougelsum: 85.8004
- Bleu: 72.7328

See: [https://wandb.ai/domenicrosati/huggingface/runs/nqf7gsws](https://wandb.ai/domenicrosati/huggingface/runs/nqf7gsws) for training and eval stats and [https://github.com/domenicrosati/qa2d-models](https://github.com/domenicrosati/qa2d-models) for the code!
## Model description

A t5-model model to convert questions, answer pairs into statements.

Due to the way it's been trained the input should be all lower case and punctuation removed.
Use with `. ` as the seperator between question and answer.
> "where in the world is carmen. abruzzo"
> Output: "carmen is in abruzzo"

Thought punctation and upper case works.

```
from transformers import AutoTokenizer,  AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('domenicrosati/QA2D-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('domenicrosati/QA2D-t5-base')

question = "where in the world is carmen sandiego"
answer = "she is in abruzzo"
SEP = ". "

prompt = f'{question}{SEP}{answer}'
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
output_ids = model.generate(input_ids)
responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
# ['carmen sandiego is in abruzzo']
```More information needed

## Intended uses & limitations

To convert questions, answer pairs into statements.

## Training and evaluation data

Uses [QA2D](https://huggingface.co/datasets/domenicrosati/QA2D).

See [https://github.com/domenicrosati/qa2d-models](https://github.com/domenicrosati/qa2d-models)


## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5.6e-05
- train_batch_size: 12
- eval_batch_size: 12
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Bleu    |
|:-------------:|:-----:|:-----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 0.2304        | 1.0   | 5060  | 0.2512          | 90.044  | 82.2922 | 85.8021 | 85.8056   | 72.6252 |
| 0.1746        | 2.0   | 10120 | 0.2525          | 90.097  | 82.3468 | 85.8191 | 85.8197   | 72.7480 |
| 0.1512        | 3.0   | 15180 | 0.2563          | 90.1064 | 82.378  | 85.7963 | 85.8004   | 72.7328 |


### Framework versions

- Transformers 4.18.0
- Pytorch 1.11.0a0+17540c5
- Datasets 2.1.0
- Tokenizers 0.12.1

"""


if __name__ == '__main__':
    candidate_metrics = extract_candidate_metrics_from_readme(readme)

    card = ModelCard(readme, ignore_metadata_errors=True)

    card_data: ModelCardData = card.data
    eval_results = card_data.eval_results
    print(card_data.metrics)

    # for eval_result in eval_results:
    #     print(eval_result.metric_name)


    # results = candidate_metrics['card_related']['results']
    # for result in results:
    #     print(result['metric_type'])