import torch
import torch.nn as nn
import torch.distributions as td

from torch_geometric.utils import to_dense_adj, to_dense_batch
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np


LATENT_DIM = 2
MAX_NODES = 28
NODE_FEATURE_DIM = 7

ENCODER_FILTER_LENGTH = 4

DECODER_HIDDEN_DIM = 512
DECODER_NUM_HIDDEN_LAYERS = 3

PRIOR_NUM_COMPONENTS = 16

class MoGPrior(nn.Module):
    """Mixture of Gaussians prior distribution."""
    def __init__(self):
        super(MoGPrior, self).__init__()
        learnable = True
        # Initialize parameters: means, variances, and mixture weights
        self.means = nn.Parameter(torch.randn(PRIOR_NUM_COMPONENTS, LATENT_DIM), requires_grad=learnable)
        self.log_vars = nn.Parameter(torch.zeros(PRIOR_NUM_COMPONENTS, LATENT_DIM), requires_grad=learnable) # Use log variance for numerical stability
        self.logits = nn.Parameter(torch.zeros(PRIOR_NUM_COMPONENTS), requires_grad=learnable)  # Mixture weights in log space
        

    def forward(self):
         # Convert log variance to actual variance using softplus
        variances = nn.functional.softplus(self.log_vars)
        # Create a mixture of Gaussians
        mixture_dist = td.Categorical(logits=self.logits)
        component_dist = td.Independent(td.Normal(self.means, variances.sqrt()), 1)
        mixture = td.MixtureSameFamily(mixture_dist, component_dist)
        return mixture


class GNNEncoder(nn.Module):
    def __init__(self):
        """
        Graph Neural Network Encoder.
        """
        super(GNNEncoder, self).__init__()
        self.mean_encoder = SimpleGraphConv()
        self.std_encoder = SimpleGraphConv()

    def forward(self, x, edge_index, batch):
        mean = self.mean_encoder(x, edge_index, batch)
        log_var = self.std_encoder(x, edge_index, batch)
        var = nn.functional.softplus(log_var)
        var[var < 1e-6] = 1e-6
        
        return td.Independent(td.Normal(loc=mean, scale=var.sqrt()), 1)


class SimpleGraphConv(nn.Module):
    """simple graph convolution block"""
    def __init__(self):
        super(SimpleGraphConv, self).__init__()

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5 * torch.randn(ENCODER_FILTER_LENGTH))
        self.h.data[0] = 1.0

        # State output network
        self.output_net = torch.nn.Linear(NODE_FEATURE_DIM, LATENT_DIM)

    def forward(self, x, edge_index, batch):
        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)

        # Implementation in spectral domain
        L, U = torch.linalg.eigh(A)
        exponentiated_L = L.unsqueeze(2).pow(
            torch.arange(ENCODER_FILTER_LENGTH, device=L.device)
        )
        diagonal_filter = (self.h[None, None] * exponentiated_L).sum(2, keepdim=True)
        node_state = U @ (diagonal_filter * (U.transpose(1, 2) @ X))

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # project to latent space
        out = self.output_net(graph_state)
        return out


class BernoulliDecoder(nn.Module):
    def __init__(self):
        super(BernoulliDecoder, self).__init__()
        """
        Bernoulli decoder network, mapping from the latent space to the upper triangular part of the adjacency matrix.
        Network architecture is a simple MLP with ReLU activations.
        """
        self.decoder_net = nn.Sequential(
            torch.nn.Linear(LATENT_DIM, DECODER_HIDDEN_DIM),
            torch.nn.ReLU(),
            *(
                [
                    torch.nn.Linear(DECODER_HIDDEN_DIM, DECODER_HIDDEN_DIM),
                    torch.nn.ReLU(),
                ]
                * DECODER_NUM_HIDDEN_LAYERS
            ),
            torch.nn.Linear(DECODER_HIDDEN_DIM, (MAX_NODES**2 - MAX_NODES) // 2),
        )

    def forward(self, z):
        return td.Independent(td.Bernoulli(logits=self.decoder_net(z)), 2)


class GraphLevelVAE(nn.Module):
    """
    Graph Variational Autoencoder (VAE) model.
    """
    def __init__(self):
        super(GraphLevelVAE, self).__init__()
        self.prior = MoGPrior()
        self.decoder = BernoulliDecoder()
        self.encoder = GNNEncoder()

    def kl_mc_approx(self, q, prior, num_samples=256):
        samples = q.rsample((num_samples,))
        log_q_x = q.log_prob(samples)
        log_prior_x = prior.log_prob(samples)
        kl_div = (log_q_x - log_prior_x).mean(0)
        return kl_div


    def elbo(self, x, edge_index, batch=None):
        q = self.encoder(x, edge_index, batch)
        z = q.rsample()

        # get the true adjacency matrix
        A = to_dense_adj(edge_index, batch) # shape: (batch_size, num_nodes, num_nodes)
        # pad the adjacency matrix to 28x28
        A = torch.nn.functional.pad(A, (0, MAX_NODES - A.shape[1], 0, MAX_NODES - A.shape[1])) # shape: (batch_size, MAX_NODES, MAX_NODES)
        # extract the upper triangular part of the adjacency matrix
        indices = torch.triu_indices(MAX_NODES, MAX_NODES, offset=1) 
        A = A[:, indices[0], indices[1]] # shape: (batch_size, (MAX_NODES**2 - MAX_NODES) // 2)

        
        kl_div = self.kl_mc_approx(q, self.prior())

        # permute the adjacency matrix ##this only worked in my old implementation there A were 28x28 matrices instead of the 378 vector
        #P = self.heristic_permutation_log_like(out.base_dist.probs.detach().numpy(), A.detach().numpy())
        #recon_loss = self.decoder(z).log_prob(torch.bmm(A,P))

        recon_loss = self.decoder(z).log_prob(A)
        elbo = torch.mean(recon_loss - kl_div, dim=0)
        return elbo

    # def heristic_permutation_log_like(self, probs, A):
    #     """assumes (probs>0.5)@P1 = A@P2
    #     and returns P = P2@P1.T
    #     such that probs = A@P

    #     where P1 and P2 are permutation matrices according to the heuristic of a depth first search on the adjacency matrix and the probabilities
    #     starting in the node with the highest degree in the adjacency matrix

    #     implementation assumes that probs is of shape (batch_size, num_nodes, num_nodes)
    #     and A is of shape (batch_size, num_nodes, num_nodes)
    #     """
    #     P1 = np.array([np.eye(probs.shape[1]) for i in range(probs.shape[0])])
    #     P2 = np.array([np.eye(probs.shape[1]) for i in range(probs.shape[0])])
        
    #     for i in range(probs.shape[0]):
    #         order = PermutationHeuristic((probs[i]>0.5).astype(float)).execute()
    #         P1[i] = P1[i][order][:, order]
    #         order = PermutationHeuristic(A[i]).execute()
    #         P2[i] = P2[i][order][:, order]

    #     # combine the two permutations
    #     P = np.array([P2[i]@P1[i].T for i in range(probs.shape[0])])
    #     P = torch.tensor(P, dtype=torch.float32)
    #     return P

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        upper_triangular = self.decoder(z).sample()
        indices = torch.triu_indices(MAX_NODES, MAX_NODES, offset=1)
        A_samples = torch.zeros(n_samples, MAX_NODES, MAX_NODES)
        A_samples[:, indices[0], indices[1]] = upper_triangular
        A_samples = A_samples + A_samples.transpose(1, 2)
        return A_samples

    def forward(self, x, edge_index, batch=None):
        return -self.elbo(x, edge_index, batch)


# class PermutationHeuristic:
#     def __init__(self, adjacency_matrix):
#         self.adj_matrix = adjacency_matrix
#         self.n = adjacency_matrix.shape[0]  
#         self.visited = np.full(self.n, False, dtype=bool)
#         self.visit_order = []

#     def dfs(self, start_node):
#         stack = [start_node]
#         while stack:
#             node = stack.pop()
#             if not self.visited[node]:
#                 self.visited[node] = True
#                 self.visit_order.append(node)
#                 adjacent_nodes = [
#                     i for i in range(self.n) if self.adj_matrix[node][i] > 0
#                 ]
#                 stack.extend(sorted(adjacent_nodes, reverse=True))

#     def execute(self):
#         degrees = self.adj_matrix.sum(axis=1)

#         while not np.all(self.visited):
#             degrees_masked = degrees * (~self.visited)
#             if np.all(degrees_masked == 0):
#                 remaining_nodes = np.where(~self.visited)[0]
#                 #returns a random permutation of the remaining nodes not connected to the rest of the graph
#                 random_order = remaining_nodes[
#                     np.random.permutation(len(remaining_nodes))
#                 ]
#                 self.visit_order.extend(random_order.tolist())
#                 break
#             next_node = np.argmax(degrees_masked)
#             self.dfs(next_node)

#         return self.visit_order
