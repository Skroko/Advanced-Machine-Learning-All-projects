# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from AMLsrc.utilities.modules import recursive_find_python_class


class DDPM(nn.Module):
    def __init__(self, config):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = recursive_find_python_class(config.network)()
        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T

        self.T = config.T

        self.beta = nn.Parameter(torch.linspace(self.beta_1, self.beta_T, self.T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###

        #sample one random time step and make it a tensor of shape (batch_size, 1)
        t = torch.randint(0, self.T, (x.shape[0], 1)).to(x.device)
        #sample epsilon from the standard normal distribution
        eps = torch.randn(x.shape).to(x.device)
        #sample x_0 from th batch of data
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1-self.alpha_cumprod[t])
        x_t = sqrt_alpha_bar_t*x+sqrt_one_minus_alpha_bar_t*eps

        eps_pred = self.network(x_t, t/self.T)
  
        #compute the objective function
        neg_elbo = torch.norm(eps - eps_pred,p=2,dim=1)**2

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            z = torch.randn(shape).to(self.alpha.device) if t > 1 else 0
            C1 = 1 / torch.sqrt(self.alpha[t])
            C2 = (1 - self.alpha[t])/torch.sqrt(1-self.alpha_cumprod[t])
            
            #make t as a tenser of shape (batch_size, 1)
            t_ = t*torch.ones(shape[0], 1).to(self.alpha.device)
            x_t = C1 *(x_t - C2 * self.network(x_t, t_/self.T)) + z*torch.sqrt(self.beta[t])
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()
