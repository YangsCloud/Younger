from huggingface_hub import list_metrics
from huggingface_hub.repocard_data import model_index_to_eval_results
import pathlib
import json
from tqdm import tqdm
from huggingface_hub import utils, HfFileSystem, ModelCard, ModelCardData, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from younger.datasets.constructors.huggingface.utils import extract_candidate_metrics_from_readme,get_huggingface_model_readmes


model_ids = pathlib.Path("/Users/zrsion/YoungBench/pengbench/Small-Neat.json")
save_dir = pathlib.Path("/Users/zrsion/YoungBench/pengbench")


def get_huggingface_metrics(save_dirpath: pathlib.Path):
    metrics = list_metrics()
    ids = [metric.id for metric in metrics]
    return ids


def get_eval_results_metric_name(eval_results, hf_metrics_metrics_name: list[str]):
    for eval_result in eval_results:
        metric = eval_result.metric_name
        metric = str(metric).lower()
        if metric not in hf_metrics_metrics_name:
            hf_metrics_metrics_name.append(metric)
            print(f'new metric name: {metric}')


def get_eval_results_metric_type(eval_results, hf_metrics_metrics_type: list[str]):
    for eval_result in eval_results:
        metric = eval_result.metric_type
        metric = str(metric).lower()
        if metric not in hf_metrics_metrics_type:
            hf_metrics_metrics_type.append(metric)
            print(f'new metric type: {metric}')


def get_metrics(card_data, hf_metrics_metrics: list[str]):
    eval_metrics = card_data.metrics
    if not eval_metrics:
        return
    for eval_metric in eval_metrics:
        metric = str(eval_metric).lower()
        if metric not in hf_metrics_metrics:
            hf_metrics_metrics.append(metric)
            print(f'new metric: {metric}')


if __name__ == '__main__':

    hf_metrics_metrics = list()
    hf_metrics_metrics_name = list()
    hf_metrics_metrics_type = list()
    combined_name_type_list = list()
    name_and_type = {}

    with open(model_ids, 'r') as file:
        data = json.load(file)

    readmes = get_huggingface_model_readmes(data)
    for readme in tqdm(readmes):
        candidate_metrics = extract_candidate_metrics_from_readme(readme)
        card = ModelCard(readme, ignore_metadata_errors=True)
        card_data: ModelCardData = card.data
        eval_results = card_data.eval_results

        get_eval_results_metric_name(eval_results, hf_metrics_metrics_name)
        get_eval_results_metric_type(eval_results, hf_metrics_metrics_type)
        get_metrics(card_data, hf_metrics_metrics)

    with open(save_dir.joinpath('hf_metrics_metrics_type.json'), 'w') as f:
        json.dump(hf_metrics_metrics_type, f, indent=4)

    with open(save_dir.joinpath('hf_metrics_metrics.json'), 'w') as f:
        json.dump(hf_metrics_metrics, f, indent=4)

    with open(save_dir.joinpath('hf_metrics_metrics_name.json'), 'w') as f:
        json.dump(hf_metrics_metrics_name, f, indent=4)

    # with open(save_dir.joinpath('combined_name_type_list.json'), 'w') as f:
    #     json.dump(combined_name_type_list, f, indent=4)