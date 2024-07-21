import torch
import torch.distributions as td
from tqdm import tqdm
import numpy as np


def iwae_bound(model, test_loader, device='cpu'):
    """
    Evaluates the Importance-Weighted Autoencoder (IWAE) bound using PyTorch, on the given test data.

    Parameters:
    - model: VAE
        The trained VAE model.
    - test_loader: DataLoader
        The test data loader.
    """
    log_weights = []

    log_q_z_given_x = lambda z, x: model.encoder(x).log_prob(z)
    log_p_x_given_z = lambda x, z: model.decoder(z).log_prob(x)
    log_p_z = lambda z: model.prior().log_prob(z)

    for x, _ in tqdm(test_loader):
        x = x.to(device)
        z = model.encoder(x).rsample()
        log_weight = log_p_x_given_z(x, z) + log_p_z(z) - log_q_z_given_x(z, x)
        log_weights.append(log_weight)

    log_weights = torch.cat(log_weights)  # Concatenate to form a tensor
    
    # Use logsumexp for numerical stability when computing the log mean of the weights
    log_mean_weight = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(log_weights.size(0), dtype=torch.float))

    return log_mean_weight

def kl_divergence_with_fallback(q, prior, num_samples=256):
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

def elbo_bound(model, test_loader, device='cpu'):
    """
    Evaluates the Evidence Lower Bound (ELBO) using PyTorch, on the given test data.

    Parameters:
    - model: VAE
        The trained VAE model.
    - test_loader: DataLoader
        The test data loader.
    """
    elbo_contributions = []

    log_p_x_given_z = lambda x, z: model.decoder(z).log_prob(x)

    for x, _ in tqdm(test_loader):
        x = x.to(device)
        q = model.encoder(x)
        z = q.rsample()
        elbo_ = log_p_x_given_z(x, z) - kl_divergence_with_fallback(q, model.prior())
        elbo_contributions.append(elbo_)
    
    elbo_contributions = torch.cat(elbo_contributions)  # Concatenate to form a tensor
    elbo = elbo_contributions.mean()
    return elbo

def get_eval_metrics(model, test_loader, device='cpu'):
    """
    Evaluates the model using PyTorch, on the given test data.

    Parameters:
    - model: VAE
        The trained VAE model.
    - test_loader: DataLoader
        The test data loader.
    """
    metrics = {}
    metrics['ELBO'] = elbo_bound(model, test_loader, device)
    metrics['IWAE'] = iwae_bound(model, test_loader, device)
    return metrics
    

    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)