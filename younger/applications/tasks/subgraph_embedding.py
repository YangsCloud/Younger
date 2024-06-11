import json
import tqdm
import pathlib
import argparse
import torch
import torch.utils.data

from torch_geometric.loader import DataLoader

from younger.commons.io import load_pickle

from younger.applications.models import GCN_NP, GAT_NP, SAGE_NP, Encoder_NP, LinearCls
from younger.applications.utils.neural_network import load_checkpoint
from younger.applications.datasets.node_dataset import NodeDataset


def get_model(baseline_model: str, encode_type: str): 
    if baseline_model == 'gcn':
        model = GCN_NP(
            node_dict_size=185 if encode_type == 'operator' else 4409,
            node_dim=1024,
            hidden_dim=512,
            dropout=0.5,
            output_embedding=True,
        )
    elif baseline_model == 'gat':
        model = GAT_NP(
            node_dict_size=185 if encode_type == 'operator' else 4409,
            node_dim=1024,
            hidden_dim=512,
            dropout=0.5,
            output_embedding=True,
        )
    elif baseline_model == 'sage':
        model = SAGE_NP(
            node_dict_size=185 if encode_type == 'operator' else 4409,
            node_dim=1024,
            hidden_dim=512,
            dropout=0.5,
            output_embedding=True,
        )
    elif baseline_model == 'gae':
        pass
    elif baseline_model == 'vgae':
        pass

    return model


def main(dataset_dir: pathlib.Path, save_dir: pathlib.Path, baseline_model: str, encode_type: str, checkpoint_filepath: pathlib.Path):
    save_dir = pathlib.Path(save_dir.joinpath(baseline_model).joinpath(encode_type))
    model = get_model(baseline_model, encode_type)
    checkpoint = load_checkpoint(checkpoint_filepath)
    model.eval()
    model.load_state_dict(checkpoint['model_state'], strict=True)

    train_dataset = NodeDataset(
        root=dataset_dir.joinpath(encode_type).joinpath('train'),
        worker_number=32,
        encode_type=encode_type,
    )
    valid_dataset = NodeDataset(
        root=dataset_dir.joinpath(encode_type).joinpath('valid'),
        worker_number=32,
        encode_type=encode_type,
    )
    test_dataset = NodeDataset(
        root=dataset_dir.joinpath(encode_type).joinpath('test'),
        worker_number=32,
        encode_type=encode_type,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_raw_dir = dataset_dir.joinpath(encode_type).joinpath('train/younger_raw_operator_np')
    valid_raw_dir = dataset_dir.joinpath(encode_type).joinpath('valid/younger_raw_operator_np')
    test_raw_dir = dataset_dir.joinpath(encode_type).joinpath('test/younger_raw_operator_np')

    total_num = 0
    hash2embedding = dict()

    with torch.no_grad():
        with tqdm.tqdm(total=len(train_dataloader)) as progress_bar:
            for index, minibatch in enumerate(train_dataloader, start=0):
                embedding = model(minibatch.x, minibatch.edge_index, minibatch.mask_x_position)
                hashing, _, _ = load_pickle(train_raw_dir.joinpath(f'sample-{index}.pkl'))
                subbgraph = {
                'embedding': embedding,
                'hash': hashing,
                }
                hash2embedding[hashing] = str(f'embedding_{total_num}')
                torch.save(subbgraph, save_dir.joinpath(f'embedding_{total_num}.pth'))
                total_num += 1
                progress_bar.update(1)

    with torch.no_grad():
        with tqdm.tqdm(total=len(valid_dataloader)) as progress_bar:
            for index, minibatch in enumerate(valid_dataloader, start=0):
                embedding = model(minibatch.x, minibatch.edge_index, minibatch.mask_x_position)
                hashing, _, _ = load_pickle(valid_raw_dir.joinpath(f'sample-{index}.pkl'))
                subbgraph = {
                'embedding': embedding,
                'hash': hashing,
                }
                hash2embedding[hashing] = str(f'embedding_{total_num}')
                torch.save(subbgraph, save_dir.joinpath(f'embedding_{total_num}.pth'))
                total_num += 1
                progress_bar.update(1)

    with torch.no_grad():
        with tqdm.tqdm(total=len(test_dataloader)) as progress_bar:
            for index, minibatch in enumerate(test_dataloader, start=0):
                embedding = model(minibatch.x, minibatch.edge_index, minibatch.mask_x_position)
                hashing, _, _ = load_pickle(test_raw_dir.joinpath(f'sample-{index}.pkl'))
                subbgraph = {
                'embedding': embedding,
                'hash': hashing,
                }
                hash2embedding[hashing] = str(f'embedding_{total_num}')
                torch.save(subbgraph, save_dir.joinpath(f'embedding_{total_num}.pth'))
                total_num += 1
                progress_bar.update(1)
        
    with open(save_dir.joinpath('hash2embedding.json'),'w') as f:
        json.dump(hash2embedding, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset-dir', type=str, required=True, help='The folder contains spilited subgraph')
    parser.add_argument('--save-dir', type=str, required=True, help='The folder to save subgraph embeddings')
    parser.add_argument('--baseline-model', type=str, required=True, help='node or operator')
    parser.add_argument('--encode-type', type=str, required=True, help='node or operator')
    parser.add_argument('--checkpoint-filepath', type=str, required=True, help='The checkpoint of chosen baseline')
    args = parser.parse_args()
    
    dataset_dir = pathlib.Path(args.dataset_dir)
    save_dir = pathlib.Path(args.save_dir)
    baseline_model = args.baseline_model
    encode_type = args.encode_type
    checkpoint_filepath = pathlib.Path(args.checkpoint_filepath)

    assert baseline_model in ['gcn', 'gat', 'sage', 'gae', 'vgae']

    main(dataset_dir, save_dir, baseline_model, encode_type, checkpoint_filepath)