from base64 import encode
import os
import json
import tqdm
import pathlib
import argparse
import networkx
import torch
import torch.utils.data
import multiprocessing

from younger.commons.io import load_pickle

from younger.datasets.modules import Instance, Network


def get_communities(graph: networkx.DiGraph, encode_type: str) -> list[tuple[networkx.DiGraph, set, str]]:
    communities = list(networkx.community.greedy_modularity_communities(graph, resolution=1, cutoff=1))

    all_subgraph_with_labels = list()
    for community in communities:
        if len(community) == 0:
            continue
        boundary = networkx.node_boundary(graph, community)
        if len(boundary) == 0:
            continue
        subgraph: networkx.DiGraph = networkx.subgraph(graph, community | boundary).copy()

        subgraph_hash = Network.hash(subgraph, node_attr='features')
        if encode_type == 'operator':
            cleansed_subgraph = networkx.DiGraph()
            cleansed_subgraph.add_nodes_from(subgraph.nodes(data=True))
            cleansed_subgraph.add_edges_from(subgraph.edges(data=True))
            for node_index in cleansed_subgraph.nodes():
                cleansed_subgraph.nodes[node_index]['operator'] = cleansed_subgraph.nodes[node_index]['features']['operator']
            subgraph_hash = Network.hash(cleansed_subgraph, node_attr='operator')

        all_subgraph_with_labels.append((subgraph, boundary, subgraph_hash))
    return all_subgraph_with_labels


def get_embedding_from_subgraph(parameters: tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, dict[str, str]]):
    graph_path, subgraph_embedding_dir, save_dir, hash2embedding = parameters
    graph = load_pickle(graph_path)    
    all_subgraph_with_labels = get_communities(graph, encode_type)
    if len(all_subgraph_with_labels) == 0:
        return
    for index, subgraph_with_label in enumerate(all_subgraph_with_labels, start=1): # subgraph_with_label = (subgraph, boundary, subgraph_hash)
        try:
            subgraph = torch.load(subgraph_embedding_dir.joinpath(f'{hash2embedding[subgraph_with_label[2]]}.pth'))
        except Exception as e:
            return

        if index == 1:
            graph_embedding = subgraph['embedding']/len(all_subgraph_with_labels)
        else: 
            graph_embedding += subgraph['embedding']/len(all_subgraph_with_labels)

        if torch.isnan(graph_embedding).any() or torch.isnan(subgraph['embedding']).any() or torch.isinf(subgraph['embedding']).any() or torch.isinf(graph_embedding).any():
            print(graph_embedding, subgraph['embedding'], len(all_subgraph_with_labels))

    processed_embeddinhg = dict(
        model_names = graph.graph['model_names'],
        graph_hash = graph.graph['hash'],
        graph_embedding = graph_embedding,
    )
    torch.save(processed_embeddinhg, save_dir.joinpath(f"{graph.graph['hash']}.pth"))


def generate_graph_embeddings(subgraph_embedding_dir: pathlib.Path, graph_dir: pathlib.Path, save_dir: pathlib.Path, config_json: pathlib.Path, block_get_type: str, worker_number: int):
    with open(config_json, 'r') as f:
        hash2embedding = json.load(f)
    if block_get_type == 'greedy_modularity_communities':
        parameters = [(graph_path, subgraph_embedding_dir, save_dir, hash2embedding) for graph_path in graph_dir.iterdir()]
        with multiprocessing.Pool(worker_number) as pool:
            with tqdm.tqdm(total=len(parameters), desc='Generating graph embeddings') as progress_bar:
                for index in enumerate(pool.imap_unordered(get_embedding_from_subgraph, parameters), start=1):
                    progress_bar.update(1)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To generate graph embedding from subgraph embeddings")
    parser.add_argument("--subgraph-dir", type=str, help="The folder contains saved subgraph embeddings")
    parser.add_argument("--graph-dir", type=str, help="The folder to load graphs")
    parser.add_argument("--save-dir", type=str, help="The folder to save graph embeddings")
    parser.add_argument("--embedding-model", type=str, default="gcn", help="The model to generater subgraph embedding")
    parser.add_argument("--encode-type", type=str, default="operator", help="The encoder-type: node or operator")
    parser.add_argument("--block-get-type", type=str, default="greedy_modularity_communities", help="The method to get communities")
    parser.add_argument("--worker-number", type=int, default=1, help="worker number")
    args = parser.parse_args()
    
    subgraph_dir = pathlib.Path(args.subgraph_dir)
    graph_dir = pathlib.Path(args.graph_dir)
    save_dir = pathlib.Path(args.save_dir)
    embedding_model = args.embedding_model
    encode_type = args.encode_type
    block_get_type = args.block_get_type
    worker_number = args.worker_number

    save_dir = pathlib.Path(save_dir.joinpath(embedding_model).joinpath(encode_type))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    subgraph_embedding_dir = subgraph_dir.joinpath(embedding_model).joinpath(encode_type)
    config_json = subgraph_embedding_dir.joinpath("hash2embedding.json")

    print(f"----now processing {embedding_model}-{encode_type}----")
    generate_graph_embeddings(subgraph_embedding_dir, graph_dir, save_dir ,config_json, block_get_type, worker_number)