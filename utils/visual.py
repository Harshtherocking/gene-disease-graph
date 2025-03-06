import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load the graph from the binary file
home = os.getcwd()
graph_path = os.path.join(home, "GRAPH")
graph, _ = dgl.load_graphs(graph_path)
g = graph[0]  # Load the first graph
print(g)

# Move graph to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = g.to(device)

# Sample a small subset of nodes (e.g., 50 nodes)
num_nodes = g.num_nodes()
subset_size = min(50, num_nodes)  # Ensure we don't exceed graph size
subset_nodes = torch.randperm(num_nodes)[:subset_size].to(device)

# Get the subgraph induced by the sampled nodes
subgraph = dgl.node_subgraph(g, subset_nodes)

# **Move the subgraph to CPU before converting to NetworkX**
subgraph_cpu = subgraph.to("cpu")
nx_graph = subgraph_cpu.to_networkx()

# Plot the graph using Matplotlib
plt.figure(figsize=(8, 6))
nx.draw(nx_graph, with_labels=True, node_size=300, node_color="skyblue", edge_color="gray")
plt.title("Small Sampled Graph Visualization")
plt.show()
