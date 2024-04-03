import json


if __name__ == '__main__':
    with open("/Users/zrsion/YoungBench/pengbench/template.json",'w') as f:
        with open("/Users/zrsion/YoungBench/pengbench/example.json", 'r') as input_file:
            data = json.load(input_file)
        json.dump(data,f, indent=4)
