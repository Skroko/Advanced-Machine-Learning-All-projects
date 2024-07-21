import networkx as nx
import torch
import numpy as np

class Evaluator:
    def __init__(self, gdl):
        self.train_graphs = gdl.train_loader.dataset
        self.empirical_hased_graphs = [
            self.hash_edge_list(g.edge_index) for g in self.train_graphs
        ]
    
    def hash_edge_list(self, edge_list):
        G = nx.Graph(edge_list.t().tolist())
        #include only largest connected component
        G = G.subgraph(max(nx.connected_components(G), key=len))
        return nx.weisfeiler_lehman_graph_hash(G)

    def evaluate(self, adj_samples):
        edge_lists = [torch.nonzero(adj).t() for adj in adj_samples]
        sample_hased_graphs = [self.hash_edge_list(edges) for edges in edge_lists]

        is_novel = torch.tensor([hash not in self.train_graphs for hash in sample_hased_graphs])
        is_unique = torch.tensor([sample_hased_graphs.count(hash) == 1 for hash in sample_hased_graphs])
        is_novel_and_unique = is_novel * is_unique

        novelty_rate = is_novel.float().mean().item()
        uniqueness_rate = is_unique.float().mean().item()
        novelty_and_uniqueness_rate = is_novel_and_unique.float().mean().item()

        return novelty_rate, uniqueness_rate, novelty_and_uniqueness_rate
    
    def node_degrees(self, adj_samples=None):
        if adj_samples is None:
            return torch.hstack([torch.hstack([(g.edge_index[0, :] == i).sum() for i in range(g.num_nodes)]) for g in self.train_graphs]).numpy()
        else:
            degrees = []
            edge_lists = [torch.nonzero(adj).t() for adj in adj_samples]
            for edges in edge_lists:
                G = nx.Graph(edges.t().tolist())
                G = G.subgraph(max(nx.connected_components(G), key=len))
                degrees.extend(list(dict(G.degree()).values()))
            return np.array(degrees)
    
    def clustering_coefficients(self, adj_samples=None):
        if adj_samples is None:
            return torch.hstack([torch.tensor(list(nx.clustering(nx.Graph(g.edge_index.t().tolist())).values())) for g in self.train_graphs]).numpy()
        else:
            coeffs = []
            edge_lists = [torch.nonzero(adj).t() for adj in adj_samples]
            for edges in edge_lists:
                G = nx.Graph(edges.t().tolist())
                G = G.subgraph(max(nx.connected_components(G), key=len))
                coeffs.extend(list(nx.clustering(G).values()))
            return np.array(coeffs)
           
    
    def eigenvector_centralities(self, adj_samples=None):
        if adj_samples is None:
            return torch.hstack([torch.tensor(list(nx.eigenvector_centrality(nx.Graph(g.edge_index.t().tolist()),max_iter=3000).values())) for g in self.train_graphs]).numpy()
        else:
            coeffs = []
            edge_lists = [torch.nonzero(adj).t() for adj in adj_samples]
            for edges in edge_lists:
                G = nx.Graph(edges.t().tolist())
                G = G.subgraph(max(nx.connected_components(G), key=len))
                coeffs.extend(list(nx.eigenvector_centrality(G,max_iter=3000).values()))
            return np.array(coeffs)