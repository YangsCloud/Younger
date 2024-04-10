from huggingface_hub import list_metrics
from huggingface_hub.repocard_data import model_index_to_eval_results
import pathlib
import json
from tqdm import tqdm
from typing import Any, Literal, Iterable, Generator, Dict
from huggingface_hub import utils, HfFileSystem, ModelCard, ModelCardData, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from younger.datasets.constructors.huggingface.utils import extract_candidate_metrics_from_readme
from younger.commons.logging import logger


model_ids = pathlib.Path("/Users/zrsion/YoungBench/pengbench/SMALL-Neat.json")
save_dir = pathlib.Path("/Users/zrsion/YoungBench/pengbench/test_data_process/lists")


def get_huggingface_model_readme(model_id: str):
    hf_file_system = HfFileSystem()
    if hf_file_system.exists(f'{model_id}/README.md'):
        try:
            with hf_file_system.open(f'{model_id}/README.md', mode='r', encoding='utf-8') as readme_file:
                readme = readme_file.read()
                readme = readme.replace('\t', ' ')
                return readme
        except UnicodeDecodeError as error:
            logger.error(
                f"REPO: {model_id}. Encoding Error - The Encoding [UTF-8] are Invalid. - Error: {error}")
            with open(save_dir.joinpath("skip.log"), 'a') as f:
                f.write(f'{model_id} failed, reason: {str(error)}\n')
            return None
        except Exception as error:
            logger.error(f"REPO: {model_id}. Encounter An Error {error}.")
            with open(save_dir.joinpath("skip.log"), 'a') as f:
                f.write(f'{model_id} failed, reason: {str(error)}\n')
            return None
    else:
        logger.info(f"REPO: {model_id}. No README.md, skip.")
        with open(save_dir.joinpath("skip.log"), 'a') as f:
            f.write(f'{model_id} failed, reason: No README.md, skip.\n')
        return None


def get_huggingface_model_readmes(model_ids) -> Generator[str, None, None]:
    hf_file_system = HfFileSystem()
    for model_id in model_ids:
        if hf_file_system.exists(f'{model_id}/README.md'):
            try:
                with hf_file_system.open(f'{model_id}/README.md', mode='r', encoding='utf-8') as readme_file:
                    readme = readme_file.read()
                    readme = readme.replace('\t', ' ')
                    yield readme
            except UnicodeDecodeError as error:
                logger.error(f"REPO: {model_id}. Encoding Error - The Encoding [UTF-8] are Invalid. - Error: {error}")
                with open(save_dir.joinpath("skip.log"), 'a') as f:
                    f.write(f'{model_id} failed, reason: {str(error)}\n')
                continue
                raise error
            except Exception as error:
                logger.error(f"REPO: {model_id}. Encounter An Error {error}.")
                with open(save_dir.joinpath("skip.log"), 'a') as f:
                    f.write(f'{model_id} failed, reason: {str(error)}\n')
                continue
                raise error
        else:
            logger.info(f"REPO: {model_id}. No README.md, skip.")
            continue


def get_huggingface_metrics():
    metrics = list_metrics()
    ids = [metric.id for metric in metrics]
    return ids


def get_eval_results_metric_name(eval_results, hf_metrics_metrics_name):
    if not eval_results:
        return
    for eval_result in eval_results:
        metric = eval_result.metric_name
        metric = str(metric).lower()
        if metric not in hf_metrics_metrics_name:
            hf_metrics_metrics_name.append(metric)
            print(f'new metric name: {metric}')


def get_eval_results_metric_type(eval_results, hf_metrics_metrics_type):
    if not eval_results:
        return
    for eval_result in eval_results:
        metric = eval_result.metric_type
        metric = str(metric).lower()
        if metric not in hf_metrics_metrics_type:
            hf_metrics_metrics_type.append(metric)
            print(f'new metric type: {metric}')


def get_metrics(card_data, hf_metrics_metrics):
    eval_metrics = card_data.metrics
    if not eval_metrics:
        return
    for eval_metric in eval_metrics:
        metric = str(eval_metric).lower()
        if metric not in hf_metrics_metrics:
            hf_metrics_metrics.append(metric)
            print(f'new metric: {metric}')


def get_merge_name_and_type(eval_results, merge_name_and_type_dic: Dict[str,int]):
    if not eval_results:
        return
    for eval_result in eval_results:
        metric_type = str(eval_result.metric_type).lower()
        metric_name = str(eval_result.metric_name).lower()
        metric = metric_name if len(metric_name) > len(metric_type) and metric_name != 'none' else metric_type
        
        if metric in merge_name_and_type_dic:
            merge_name_and_type_dic[metric] += 1
        else:
            merge_name_and_type_dic[metric] = 1


def get_combined_name_type(eval_results, combined_name_type_list):
    if not eval_results:
        return
    for eval_result in eval_results:
        type = str(eval_result.metric_type).lower()
        name = str(eval_result.metric_name).lower()
        name_and_type = {
            "name": name,
            "type": type
        }
        final_dic = {
            "name_and_type": name_and_type,
            "count": 1
        }
        if len(combined_name_type_list) == 0:
            combined_name_type_list.append(final_dic)
        else:
            for d in combined_name_type_list:
                if name_and_type == d["name_and_type"]:
                    d["count"] += 1
                else:
                    combined_name_type_list.append(final_dic)
                    # print(final_dic)
                    break


def get_full_list(model_name, eval_results, full_list):
    if not eval_results:
        return
    for eval_result in eval_results:
        metric_name = str(eval_result.metric_name).lower()
        metric_type = str(eval_result.metric_type).lower()
        task_name = str(eval_result.task_name).lower()
        task_type = str(eval_result.task_type).lower()
        dataset_name = str(eval_result.dataset_name).lower()
        dataset_type = str(eval_result.dataset_type).lower()

        task = task_name if len(task_name) > len(task_type) and task_name != 'none' else task_type
        dataset = dataset_name if len(dataset_name) > len(dataset_type) and dataset_name != 'none' else dataset_type
        metric = metric_name if len(metric_name) > len(metric_type) and metric_name != 'none' else metric_type
        value = eval_result.metric_value if eval_result else 'none'

        task_dataset_metric_value = {
            "model_name": model_name,
            "metric": metric,
            "task": task,
            "dataset": dataset,
            "value": value
        }

        full_list.append(task_dataset_metric_value)



def get_combined_metric_task_dataset(eval_results, combined_metric_task_dataset_list):
    if not eval_results:
        return
    for eval_result in eval_results:
        metric_name = str(eval_result.metric_name).lower()
        metric_type = str(eval_result.metric_type).lower()
        task_name = str(eval_result.task_name).lower()
        task_type = str(eval_result.task_type).lower()
        dataset_name = str(eval_result.dataset_name).lower()
        dataset_type = str(eval_result.dataset_type).lower()

        task = task_name if len(task_name) > len(task_type) and task_name != 'none' else task_type
        dataset = dataset_name if len(dataset_name) > len(dataset_type) and dataset_name != 'none' else dataset_type
        metric = metric_name if len(metric_name) > len(metric_type) and metric_name != 'none' else metric_type

        task_dataset_metric = {
            "task": task,
            "dataset": dataset,
            "metric": metric
        }
        final_dic = {
            "task_dataset_metric": task_dataset_metric,
            "count": 1
        }

        if len(combined_metric_task_dataset_list) == 0:
            combined_metric_task_dataset_list.append(final_dic)
        else:
            for d in combined_metric_task_dataset_list:
                if task_dataset_metric == d["task_dataset_metric"]:
                    d["count"] += 1
                else:
                    combined_metric_task_dataset_list.append(final_dic)
                    # print(final_dic)
                    break


def get_combined_metric_task(eval_results, combined_metric_task_list):
    if not eval_results:
        return
    for eval_result in eval_results:
        metric_name = str(eval_result.metric_name).lower()
        metric_type = str(eval_result.metric_type).lower()
        task_name = str(eval_result.task_name).lower()
        task_type = str(eval_result.task_type).lower()

        task = task_name if len(task_name) > len(task_type) and task_name != 'none' else task_type
        metric = metric_name if len(metric_name) > len(metric_type) and metric_name != 'none' else metric_type

        task_metric = {
            "metric": metric,
            "task": task
        }

        final_dic = {
            "task_metric": task_metric,
            "count": 1
        }

        if len(combined_metric_task_list) == 0:
            combined_metric_task_list.append(final_dic)
        else:
            for d in combined_metric_task_list:
                if task_metric == d["task_metric"]:
                    d["count"] += 1
                else:
                    combined_metric_task_list.append(final_dic)
                    # print(final_dic)
                    break


def get_combined_metric_dataset(eval_results, combined_metric_dataset_list):
    if not eval_results:
        return
    for eval_result in eval_results:
        metric_name = str(eval_result.metric_name).lower()
        metric_type = str(eval_result.metric_type).lower()
        dataset_name = str(eval_result.dataset_name).lower()
        dataset_type = str(eval_result.dataset_type).lower()

        dataset = dataset_name if len(dataset_name) > len(dataset_type) and dataset_name != 'none' else dataset_type
        metric = metric_name if len(metric_name) > len(metric_type) and metric_name != 'none' else metric_type

        dataset_metric = {
            "dataset": dataset,
            "metric": metric
        }

        final_dic = {
            "dataset_metric": dataset_metric,
            "count": 1
        }

        if len(combined_metric_dataset_list) == 0:
            combined_metric_dataset_list.append(final_dic)
        else:
            for d in combined_metric_dataset_list:
                if dataset_metric == d["dataset_metric"]:
                    d["count"] += 1
                else:
                    combined_metric_dataset_list.append(final_dic)
                    # print(final_dic)
                    break



if __name__ == '__main__':
    full_list = list()
    hf_metrics_metrics = list()
    hf_metrics_metrics_name = list()
    hf_metrics_metrics_type = list()
    merge_name_and_type_dic = dict()
    combined_name_type_list = list()
    combined_metric_task_dataset_list = list()
    combined_metric_task_list = list()
    combined_metric_dataset_list = list()

    with open(model_ids, 'r') as file:
        data = json.load(file)
    #
    # readmes = get_huggingface_model_readmes(data)
    for model_id in tqdm(data):
        print(model_id)
        readme = get_huggingface_model_readme(model_id)
        if not readme:
            continue
        card = ModelCard(readme, ignore_metadata_errors=True)
        card_data: ModelCardData = card.data
        eval_results = card_data.eval_results
        try:
            get_full_list(model_id, eval_results, full_list)
            # get_eval_results_metric_name(eval_results, hf_metrics_metrics_name)
            # get_eval_results_metric_type(eval_results, hf_metrics_metrics_type)
            # get_metrics(card_data, hf_metrics_metrics)
            get_merge_name_and_type(eval_results, merge_name_and_type_dic)
            get_combined_name_type(eval_results, combined_name_type_list)
            get_combined_metric_task_dataset(eval_results, combined_metric_task_dataset_list)
            get_combined_metric_task(eval_results, combined_metric_task_list)
            get_combined_metric_dataset(eval_results, combined_metric_dataset_list)

        except Exception as error:
            with open(save_dir.joinpath("skip.log"), 'a') as f:
                f.write(f'{card_data.model_name} failed, reason: {str(error)}\n')
            continue

    # with open(save_dir.joinpath('hf_list_metrics.json'), 'w') as f:
    #     json.dump(get_huggingface_metrics(), f, indent=4)

    with open(save_dir.joinpath('merge_name_and_type.json'), 'w') as f:
        json.dump(merge_name_and_type_dic, f, indent=4)

    with open(save_dir.joinpath('full_list.json'), 'w') as f:
        json.dump(full_list, f, indent=4)

    # with open(save_dir.joinpath('hf_metrics_metrics_type.json'), 'w') as f:
    #     json.dump(hf_metrics_metrics_type, f, indent=4)

    # with open(save_dir.joinpath('hf_metrics_metrics.json'), 'w') as f:
    #     json.dump(hf_metrics_metrics, f, indent=4)

    # with open(save_dir.joinpath('hf_metrics_metrics_name.json'), 'w') as f:
    #     json.dump(hf_metrics_metrics_name, f, indent=4)

    with open(save_dir.joinpath('combined_name_type_list.json'), 'w') as f:
        json.dump(combined_name_type_list, f, indent=4)

    with open(save_dir.joinpath('combined_metric_task_dataset_list.json'), 'w') as f:
        json.dump(combined_metric_task_dataset_list, f, indent=4)

    with open(save_dir.joinpath('combined_metric_task_list.json'), 'w') as f:
        json.dump(combined_metric_task_list, f, indent=4)

    with open(save_dir.joinpath('combined_metric_dataset.json'), 'w') as f:
        json.dump(combined_metric_dataset_list, f, indent=4)