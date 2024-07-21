import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm

class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        # Apply mask
        z_masked = self.mask * z
        
        # Scale and translate the unmasked part
        s = self.scale_net(z_masked)
        t = self.translation_net(z_masked)
        
        # Transform
        z_transformed = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        
        # Compute log determinant
        log_det_J = torch.sum(s * (1 - self.mask), dim=-1)
        
        return z_transformed, log_det_J

    def inverse(self, z):

        # Apply mask along the feature dimension
        z_masked = self.mask * z
        
        # Scale and translate the unmasked part
        s = self.scale_net(z_masked)
        t = self.translation_net(z_masked)
        
        # Inverse transform
        z_transformed = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
        
        # Compute log determinant for the inverse
        log_det_J = -torch.sum(s * (1 - self.mask), dim=-1)
        return z_transformed, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=1):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample((sample_shape,))
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))
    


class MNISTFlow(Flow):
    def __init__(self, config):
        """
        Define a flow model for MNIST data.
        
        Parameters:
        """
        self.dim = config.dim #input dimension 28*28=784
        self.num_transformations = config.num_transformations
        self.num_hidden = config.num_hidden

        #get a mask
        self.masking = config.masking
        mask = self.get_mask(self.masking)

        # Define base distribution
        base = GaussianBase(self.dim)

        # Define transformations
        transformations =[]
        for i in range(self.num_transformations):
            mask = (1-mask) # Flip the mask
            scale_net = nn.Sequential(nn.Linear(self.dim, self.num_hidden), nn.ReLU(), nn.Linear(self.num_hidden, self.dim), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(self.dim, self.num_hidden), nn.ReLU(), nn.Linear(self.num_hidden, self.dim))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        
        transformations = nn.ModuleList(transformations)
        super(MNISTFlow, self).__init__(base, transformations)
        

    def get_mask(self, type = 'checkerboard'):
        if type == 'checkerboard':
            mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
        return mask





# def get_flow_model_mnist(D, device='cpu',masking = 'checkerboard'):
#     # Define prior distribution
#     base = GaussianBase(D)

#     # Define transformations
#     transformations =[]
    
#     num_transformations = 20
#     num_hidden = 128

#     if masking == 'checkerboard':
#         mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
#     for i in range(num_transformations):
#         if masking == 'checkerboard':
#             mask = (1-mask) # Flip the mask
#         elif masking == 'random':
#             mask = torch.Tensor([1 if torch.rand(1) > 0.5 else 0 for i in range(28) for j in range(28)])
#         else:
#             raise NotImplementedError(f"Masking {masking} not implemented")

#         scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
#         translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
#         transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

#     # Define flow model
#     model = Flow(base, transformations).to(device)
#     return model