import os
import dgl
import torch
import pandas as pd
from collections import defaultdict

class NameEncoder:
    """
    Encodes disease and gene names into unique integers and supports reverse mapping.
    """
    def __init__(self):
        self.name_to_id = {}
        self.id_to_name = {}
        self.counter = 0

    def encode(self, name):
        if name not in self.name_to_id:
            self.name_to_id[name] = self.counter
            self.id_to_name[self.counter] = name
            self.counter += 1
        return self.name_to_id[name]

    def decode(self, idx):
        return self.id_to_name.get(idx, None)

    def get_mapping(self):
        return self.name_to_id

class GraphLoader:
    """
    Loads a dataset from a file, processes it, and converts it into a DGL graph.
    """
    def __init__(self, input_path, output_path):
        self.project_dir = os.getcwd()
        self.input_path = os.path.join(self.project_dir, input_path)
        self.output_path = os.path.join(self.project_dir, output_path)
        self.encoder = NameEncoder()

    def load_data(self):
        """ Reads the data file and encodes diseases and genes. """
        df = pd.read_csv(self.input_path, sep='\t')
        
        # Assuming columns are 'Disease(MESH)' and 'Gene'
        df.columns = ['disease', 'gene']
        df['disease_id'] = df['disease'].apply(self.encoder.encode)
        df['gene_id'] = df['gene'].apply(self.encoder.encode)
        
        return df
    
    def create_graph(self, df):
        """ Creates a DGL graph from the processed data. """
        src_nodes = df['disease_id'].tolist()
        dst_nodes = df['gene_id'].tolist()
        
        # Create a DGL graph
        graph = dgl.graph((src_nodes, dst_nodes))
        graph.ndata['ids'] = torch.tensor(list(self.encoder.get_mapping().values()))
        
        return graph
    
    def save_graph(self, graph):
        """ Saves the DGL graph to the output path. """
        dgl.save_graphs(self.output_path, [graph])
    
    def process(self):
        """ Runs the complete pipeline: loading data, creating graph, and saving it. """
        df = self.load_data()
        graph = self.create_graph(df)
        self.save_graph(graph)
        print(f"Graph saved successfully to {self.output_path}")

if __name__ == "__main__":
    input_file = "gene-disease-graph/DG-Miner_miner-disease-gene.tsv"  # Example input path
    output_file = "output/graph.bin"  # Example output path
    graph_loader = GraphLoader(input_file, output_file)
    graph_loader.process()
