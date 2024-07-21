import torch


class ErdosRenyi:
    def __init__(self, gdl):
        self.train_graphs = gdl.train_loader.dataset
        self.dist_num_nodes = [g.num_nodes for g in self.train_graphs]
        self.max_nodes = max([g.num_nodes for g in self.train_graphs])
        self.link_probabilities = {}
        for n in set(self.dist_num_nodes):
            num_edges = [g.num_edges for g in self.train_graphs if g.num_nodes == n]
            mean_num_edges = sum(num_edges) / len(num_edges)
            self.link_probabilities[n] = mean_num_edges / (n * (n - 1)) #num_edges count each edge twice so n*(n-1) is the number of possible edges

    def sample_num_nodes(self):
        rand_idx = torch.randint(0, len(self.train_graphs), (1,))
        return self.dist_num_nodes[rand_idx]

    def generate_adjacency_matrix(self, num_nodes, link_probability):
        # Create an adjacency matrix where each entry is 1 with a probability of link_probability
        idx = torch.triu_indices(num_nodes, num_nodes, 1)
        adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        adjacency_matrix[idx[0], idx[1]] = torch.bernoulli(
            link_probability * torch.ones((num_nodes**2 - num_nodes) // 2)
        )
        # Ensure the adjacency matrix is symmetric and has zeros on the diagonal
        adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
        return adjacency_matrix

    def sample(self, num_graphs=1, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        samples = []
        for _ in range(num_graphs):
            n = self.sample_num_nodes()
            adj = self.generate_adjacency_matrix(n, self.link_probabilities[n])
            samples.append(adj)
        return samples
