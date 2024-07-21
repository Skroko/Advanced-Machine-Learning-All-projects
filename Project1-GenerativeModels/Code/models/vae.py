import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from AMLsrc.models.flow import *

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)



class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        #self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)
    

class ContinuesBernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(ContinuesBernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        #self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.ContinuousBernoulli(logits=logits), 2)
    
   
class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net, constant_std=True):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.constant_std = constant_std
        self.feature_dim = (28, 28)
        if constant_std:
            print("Using a constant std")
            self.log_std = nn.Parameter(torch.randn(1), requires_grad=True)
        else:
            print("Using a non constant std")
            self.log_std = nn.Parameter(torch.zeros(28, 28), requires_grad=True)

    def forward(self, z):
        """
        Forward pass to return a Gaussian distribution over the data space given latent variables.

        Parameters:
        z (torch.Tensor): Latent variable tensor of dimension (batch_size, M).

        Returns:
        torch.distributions.Distribution: A Gaussian distribution parameterized by the decoder output.
        """
        mean = self.decoder_net(z)
        return td.Independent(td.Normal(mean, torch.exp(self.log_std)), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder


    def kl_divergence_with_fallback(self, q, prior, num_samples=256):
        try:
            # Attempt to directly calculate KL divergence
            kl_div = td.kl_divergence(q, prior)
        except NotImplementedError:
            # Fallback to Monte Carlo estimation if direct calculation fails
            samples = q.rsample((num_samples,))
            log_q_x = q.log_prob(samples)
            log_prior_x = prior.log_prob(samples)
            kl_div = (log_q_x - log_prior_x).mean(0)
        return kl_div


    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - self.kl_divergence_with_fallback(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    def sample_from_prior_show_mean_of_px_given_z(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).mean
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """        
        return -self.elbo(x)

class AddChannelDim(nn.Module):
    def __init__(self, dim):
        super(AddChannelDim, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Unsqueeze the specified dimension
        return x.unsqueeze(self.dim)

class SqueezeChannelDim(nn.Module):
    def __init__(self, dim):
        super(SqueezeChannelDim, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Squeeze the specified dimension
        return x.squeeze(self.dim)



class LinearEncoderDecoderNets:
    def __init__(self, latent_dim,unflatten=True):
        """
        Create a simple linear encoder and decoder network.

        Parameters:
        """
        self.D = 784
        self.encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, latent_dim * 2),
        )

        self.decoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
        )
        if unflatten:
            self.decoder_net.add_module('unflatten', nn.Unflatten(-1, (1, 28, 28)))
    def __call__(self):
        return self.encoder_net, self.decoder_net

class CNNEncoderDecoderNets:
    def __init__(self, latent_dim):
        """
        Create a simple CNN encoder and decoder network.

        Parameters:
        """
        self.D = 784
        self.encoder_net = nn.Sequential(
        AddChannelDim(dim=1),
        nn.Conv2d(1, 16, 5),
        nn.ReLU(),
        nn.Conv2d(16, 32, 5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*20*20, latent_dim * 2),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 32*20*20),
            nn.ReLU(),
            nn.Unflatten(-1, (32, 20, 20)),
            nn.ConvTranspose2d(32, 16, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 5),
            SqueezeChannelDim(dim=1),
        )
    def __call__(self):
        return self.encoder_net, self.decoder_net

