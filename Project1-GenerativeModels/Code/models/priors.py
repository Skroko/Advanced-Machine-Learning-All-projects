import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from AMLsrc.models.flow import *
import numpy as np

PI = torch.from_numpy(np.asarray(np.pi))

class GaussianPrior(nn.Module):
    def __init__(self, M, **kwargs):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianMixturePrior(nn.Module):
    def __init__(self, dim, **kwargs):
        super(GaussianMixturePrior, self).__init__()
        self.dim = dim
        self.num_components = 16 # Number of components in the mixture

        # Initialize parameters: means, variances, and mixture weights
        self.means = nn.Parameter(torch.randn(self.num_components, dim), requires_grad=True)
        self.log_vars = nn.Parameter(torch.zeros(self.num_components, dim), requires_grad=True) # Use log variance for numerical stability
        self.logits = nn.Parameter(torch.zeros(self.num_components), requires_grad=True)  # Mixture weights in log space
        

    def forward(self):
         # Convert log variance to actual variance
        variances = torch.exp(self.log_vars)
        
        # Create a mixture of Gaussians
        mixture_dist = td.Categorical(logits=self.logits)
        component_dist = td.Independent(td.Normal(self.means, variances.sqrt()), 1)
        mixture = td.MixtureSameFamily(mixture_dist, component_dist)
        return mixture
    

class FlowPrior(nn.Module):
    def __init__(self,dim,**kwargs):
        """
        Define a flow prior distribution.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        num_flows: [int] 
           Number of coupling layers to use in the flow.
        flow_type: [str] 
           Type of flow to use. Currently only "maf" is supported.
        """
        super(FlowPrior, self).__init__()
        self.dim = dim
        self.masking = kwargs.get('masking', 'checkerboard')
        # Define the base distribution
        self.base_dist = GaussianBase(dim)

        self.num_transforms = 10
        self.num_hidden = 8
        self.transformations =[]

        #if masking == 'checkerboard':
        mask = torch.Tensor([1 if (i) % 2 == 0 else 0 for i in range(dim)])
        for i in range(self.num_transforms):
            if self.masking == 'checkerboard':
                mask = (1-mask) # Flip the mask
            elif self.masking == 'random':
                mask = torch.Tensor([1 if torch.rand(1) > 0.5 else 0 for i in range(dim)])
            else:
                raise NotImplementedError(f"Masking {self.masking} not implemented")
        scale_net = nn.Sequential(nn.Linear(dim, self.num_hidden), nn.ReLU(), nn.Linear(self.num_hidden, dim))
        translation_net = nn.Sequential(nn.Linear(dim, self.num_hidden), nn.ReLU(), nn.Linear(self.num_hidden, dim))
        self.transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

        # Define flow model
        self.flow = Flow(self.base_dist, self.transformations)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return self.flow


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p
    

class VampPrior(nn.Module):
    def __init__(self, dim, **kwargs):
        super(VampPrior, self).__init__()
        self.dim = dim # latent dimension
        self.num_components = 16 # Number of components in the mixture
        self.encoder = kwargs.get('encoder', None) # this means that the encoder is a part of the prior
        self.D = self.encoder.encoder_net[1].in_features # input dimension of the encoder
        
        # pseudo-inputs
        self.u = nn.Parameter(torch.rand(self.num_components, self.D), requires_grad=True)
        
        
        # mixing weights
        self.logits = nn.Parameter(torch.zeros(self.num_components), requires_grad=True)
        
    def forward(self):
        mixture_dist = td.Categorical(logits=self.logits)
        component_dist = self.encoder(self.u) #get encoder distribution for pseudo-inputs
        mixture = td.MixtureSameFamily(mixture_dist, component_dist)
        return mixture


