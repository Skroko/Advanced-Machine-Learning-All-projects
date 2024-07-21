from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torchvision import transforms

from enum import Enum

class Sorting(Enum):
    NONE = 0
    DEGREE = 1
    SPECTRAL = 2
    BFS = 3

import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import torch_geometric
class SpectralSorting:

    def __call__(self, data):
        G = to_networkx(data, to_undirected=True)
        laplacian = nx.normalized_laplacian_matrix(G).todense()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        fiedler_vector = eigenvectors[:, 1]
        permutation = torch.tensor(fiedler_vector.argsort(), device=data.x.device)
        data.x = data.x[permutation]
        data.edge_index, _ = torch_geometric.utils.subgraph(permutation, data.edge_index, relabel_nodes=True)
        return data

class GraphDataLoader:
    def __init__(self, dataset_name, batch_size=(100,44,44), device='cpu', seed=0, sorting: Sorting = Sorting.NONE):
        self.dataset_name = dataset_name
        self.batch_size = batch_size # (train, validation, test)
        self.total_num_graphs = sum(batch_size)
        self.device = device
        self.seed = seed
        self.setup_data_loaders()


    def setup_data_loaders(self):
        match self.dataset_name:
            case 'MUTAG':
                self.setup_MUTAG_data_loaders()
            case _:
                raise ValueError(f'Unknown dataset: {self.dataset_name}')

    def setup_MUTAG_data_loaders(self):
        # Load data
        self.dataset = TUDataset(root='./data/', name='MUTAG', transform=SpectralSorting()).to(self.device)
        self.node_feature_dim = 7

        # Split into training and validation
        rng = torch.Generator().manual_seed(self.seed)
        if self.batch_size[2] == 0:
            train_dataset, validation_dataset = random_split(self.dataset, self.batch_size[0:2], generator=rng)
        else:
            train_dataset, validation_dataset, test_dataset = random_split(self.dataset, self.batch_size, generator=rng)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size[2])
            self.n_test = len(test_dataset)

        # Create dataloader for training and validation
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size[0])
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size[1])
        

        self.node_labels = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
        self.edge_labels = {0: 'aromatic', 1: 'single', 2: 'double', 3: 'triple'}
    
        self.n_train = len(train_dataset)
        self.n_val = len(validation_dataset)

        self.color_dict = {
            0: "Red",
            1: "Blue",
            2: "Yellow",
            3: "Green",
            4: "Gray",
            5: "Purple",
            6: "Pink",
            }
    
    def plot_graphs(self, graph_list, num_graphs=10, title="MUTAG", seed=0, figsize=(30, 5)):
        perm = torch.randperm(len(graph_list), generator=torch.Generator().manual_seed(seed))
        graphs_nodes = []
        node = {}
        edges = []
        n_i = 0
        for i in range(num_graphs):
            graph = graph_list[perm[i]]
            graph_nodes = []
            for j in range(graph.num_nodes):
                node[str(j+n_i)] = dict(color=self.color_dict[torch.argmax(graph.x[j]).item()])
                graph_nodes.append(str(j+n_i))
            for k in range(graph.num_edges):
                edges.append(
                    (
                        str(graph.edge_index[0, k].item()+n_i),
                        str(graph.edge_index[1, k].item()+n_i),
                        self.edge_labels[torch.argmax(graph.edge_attr[i]).item()],
                    )
                )
            n_i += graph.x.shape[0]
            graphs_nodes.append(graph_nodes)

        original_graph = nx.Graph()
        # Add nodes and edges
        original_graph.add_nodes_from(n for n in node.items())
        original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)

        # Identify connected components
        # components = list(nx.connected_components(original_graph))

        # Initialize layout dictionary
        pos = {}
        x_offset = 1.5  # Starting x offset
        y_offset = -1  # Starting y offset
        x_shift = 0.5  # Spacing between components
        y_shift = 2.0  # Spacing between components
        y_shift_active = 1.0  # Spacing between nodes in the same component

        # Calculate layout for each component
        new_graphs= []
        for graph in graphs_nodes:
            subgraph = original_graph.subgraph(graph)
            #keep the largest connected component
            sub_pos = nx.spring_layout(subgraph, seed=7482934, iterations=500)  # Consistent layout with seed

            # Adjust position with an x offset
            for node, (x, y) in sub_pos.items():
                pos[node] = (x + x_offset, y + y_offset)

            # Calculate the width of the current component to determine the next offset
            x_width = max(x for _, x in sub_pos.values()) - min(x for _, x in sub_pos.values())
            x_offset += x_width/2 + x_shift
            y_offset += y_shift*y_shift_active
            y_shift_active *= -1
            new_graphs.append(subgraph.nodes)



        # Set up the plot
        plt.figure(figsize=figsize, dpi=300)
        base_options = dict(with_labels=False, edgecolors="black", node_size=200)

        # Define node colors
        node_colors = [d["color"] for _, d in original_graph.nodes(data=True)]

        # Set edge widths based on type
        edge_type_visual_weight_lookup = {
            "aromatic": 1,
            "single": 3,
            "double": 5,
            "triple": 7,
        }
        edge_weights = [edge_type_visual_weight_lookup[d["type"]] for _, _, d in original_graph.edges(data=True)]

        # Draw the graph
        nx.draw_networkx(
            original_graph,
            pos=pos,
            node_color=node_colors,
            width=edge_weights,
            **base_options
        )
        plt.tight_layout()
        # annotate text vertically
        plt.text(0, 0, title, fontsize=36, fontweight="bold", color="black", ha="center", va="center")
        plt.axis("off")


    def plot_graphs_from_adjacency_matrix(self, adjacency_matrix_list, num_graphs=10, title="MUTAG", seed=0,figsize=(30, 5)):
        graph_list = [GraphFromAdjacencyMatrix(adjacency_matrix) for adjacency_matrix in adjacency_matrix_list]
        self.plot_graphs(graph_list, num_graphs, title, seed, figsize=figsize)

class GraphFromAdjacencyMatrix():
    def __init__(self, adjacency_matrix):
        adjacency_matrix = self.keep_only_largest_connected_component(adjacency_matrix)
        self.num_nodes = adjacency_matrix.shape[0]
        self.num_edges = int(adjacency_matrix.sum().item())
        self.edge_index = torch.nonzero(adjacency_matrix).t()
        self.edge_attr = torch.ones(self.num_edges+100)
        self.x = torch.ones(self.num_nodes)
    
    def keep_only_largest_connected_component(self, adjacency_matrix):
        self.edge_index = torch.nonzero(adjacency_matrix)
        G = nx.Graph(self.edge_index.tolist())
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        G = G.subgraph(largest_component)
        A= nx.adjacency_matrix(G)
        A = A.todense()
        A = torch.tensor(A)
        return A

