from youngbench.dataset.modules.instance import Instance
import argparse
import pathlib
import json


def count_in_flg_files(folder_path):
    folder = pathlib.Path(folder_path)
    # print(folder)
    total_succ = 0
    total_fail = 0

    for file_path in folder.glob("*.flg"):
        # print("yes")
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # print(data)
                    if data["mode"]=="succ":
                        total_succ += 1
                    else:
                        total_fail += 1
                except json.JSONDecodeError:
                    pass

    return total_succ,total_fail

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('--instance-dir', type=pathlib.Path, default=None)
    args = parser.parse_args()

    print(f'total succ/fail : {count_in_flg_files(args.instance_dir)}')





