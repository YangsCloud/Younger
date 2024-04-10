import json
from pengbench.metric_to_target import get_target_metric
from younger.datasets.utils.metric_cleaner import clean_metric

if __name__ == '__main__':
    metrics_path = "/Users/zrsion/YoungBench/pengbench/test_data_process/mapping/metrics.json"
    metric_name = "recall_at_7"
    metric_type = "Recall"

    with open("/Users/zrsion/YoungBench/pengbench/test_data_process/combined_name_type_list.json",'r') as f:
        datas = json.load(f)

    for data in datas:
        metric_name = data['name']
        metric_name = data['type']
        print(
            f'target : {clean_metric(metric_name=metric_name,metric_type=metric_type)}'
        )
