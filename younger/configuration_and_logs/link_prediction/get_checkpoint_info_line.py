import re
import json
import pathlib
import argparse

def extract_and_average_metrics(log_file_path, output_json_path):
    #"Overall Result - [AUC]=<value> [F1]=<value> [AP]=<value>"
    pattern = re.compile(r"Overall Result - \[AUC\]=([\d.]+) \[F1\]=([\d.]+) \[AP\]=([\d.]+)")

    results = {}
    max_average = None
    max_average_line = None

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if match:
            auc = float(match.group(1))
            f1 = float(match.group(2))
            ap = float(match.group(3))
            average = (auc + f1 + ap) / 3
            results[line_number] = average
            if max_average is None or average > max_average:
                    max_average = average
                    max_average_line = line_number
    results['max_average_line'] = max_average_line
    save_name = log_file_path.stem + '_info_line.json'
    with open(save_dir.joinpath(save_name), 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results have been saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--log-file-path', type=str, required=True, help='The folder contains spilited subgraph')
    parser.add_argument('--save-dir', type=str, required=True, help='The folder to save subgraph embeddings')
    args = parser.parse_args()
    log_file_path = pathlib.Path(args.log_file_path)
    save_dir = pathlib.Path(args.save_dir)

    extract_and_average_metrics(log_file_path, save_dir)
