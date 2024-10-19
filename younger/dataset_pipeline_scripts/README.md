### NOTE

If one want to get the split dataset 'operator w/o attributes', must uncomment the lines: [here](younger/datasets/constructors/official/community_split.py)

Change:

```
# cleansed_subgraph = networkx.DiGraph()
# cleansed_subgraph.add_nodes_from(subgraph.nodes(data=True))
# cleansed_subgraph.add_edges_from(subgraph.edges(data=True))
# for node_index in cleansed_subgraph.nodes():
#     cleansed_subgraph.nodes[node_index]['operator'] = cleansed_subgraph.nodes[node_index]['features']['operator']
# subgraph_hash = Network.hash(cleansed_subgraph, node_attr='operator')
```

To:

```
cleansed_subgraph = networkx.DiGraph()
cleansed_subgraph.add_nodes_from(subgraph.nodes(data=True))
cleansed_subgraph.add_edges_from(subgraph.edges(data=True))
for node_index in cleansed_subgraph.nodes():
    cleansed_subgraph.nodes[node_index]['operator'] = cleansed_subgraph.nodes[node_index]['features']['operator']
subgraph_hash = Network.hash(cleansed_subgraph, node_attr='operator')
```
