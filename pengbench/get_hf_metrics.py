from huggingface_hub import list_metrics
from huggingface_hub.repocard_data import model_index_to_eval_results
import pathlib
import json
from huggingface_hub import utils, HfFileSystem, ModelCard, ModelCardData, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from younger.datasets.constructors.huggingface.utils import extract_candidate_metrics_from_readme,get_huggingface_model_readmes


model_ids = pathlib.Path("/Users/zrsion/YoungBench/pengbench/20K-Neat.json")
save_dir = pathlib.Path("/Users/zrsion/YoungBench/pengbench")

def get_huggingface_metrics(save_dirpath: pathlib.Path):
    metrics = list_metrics()
    ids = [metric.id for metric in metrics]
    return ids

def get_eval_results_metric_name(readme: str, metrics_list: list[str]):
    candidate_metrics = extract_candidate_metrics_from_readme(readme)
    card = ModelCard(readme, ignore_metadata_errors=True)
    card_data: ModelCardData = card.data
    eval_results = card_data.eval_results
    for eval_result in eval_results:
        metric_name = eval_result.metric_name
        metric_name = str(metric_name).lower()
        if metric_name not in metrics_list:
            metrics_list.append(metric_name)
            print(metric_name)



if __name__ == '__main__':

    # metrics = list_metrics()
    metrics_list = list()

    with open('20K-Neat.json', 'r') as file:
        data = json.load(file)

    readmes = get_huggingface_model_readmes(data)
    for readme in readmes:
        get_eval_results_metric_name(readme, metrics_list)

    with open(save_dir.joinpath('metrics.json'), 'w') as f:
        json.dump(metrics_list, f, indent=4)







