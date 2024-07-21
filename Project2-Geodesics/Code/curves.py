import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch.distributions as td


class ThirdOrderPolynomialCurve(nn.Module):
    """Defines a third order polynomial curve through two points, x0 and x3, parameterized by t in [0, 1].
    We will use cubic Bezier curve to define the curve.
    
    Args:
        x0 (torch.tensor): First point
        x3 (torch.tensor): Fourth point (end point)
    """
    
    def __init__(self, x0: torch.tensor, x1: torch.tensor):
        super(ThirdOrderPolynomialCurve, self).__init__()
        self.x0 = x0
        self.x1 = x1
        # Initialize control points P1 and P2 as torch parameters to be optimized
        # Initially setting them to be linearly spaced between x0 and x3
        self.P1 = nn.Parameter((2*x0 + x1) / 3, requires_grad=True)
        self.P2 = nn.Parameter((x0 + 2*x1) / 3, requires_grad=True)
    
    def get_points(self):
        return torch.cat([self.x0[:, None], self.P1[:, None], self.P2[:, None], self.x1[:, None]], dim=1)
    def set_control_points(self, P1: torch.tensor, P2: torch.tensor):
        """manually set the control points
        Args:
            control_points (torch.tensor): control points
        """
        self.P1.data = P1
        self.P2.data = P2

    def randomly_update_control_points(self):
        self.P1.data = torch.randn_like(self.P1.data)
        self.P2.data = torch.randn_like(self.P2.data)
    
    def evaluate(self, t: torch.tensor) -> torch.tensor:
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        return ((1-t)**3)*self.x0 + 3*((1-t)**2)*t*self.P1 + 3*(1-t)*(t**2)*self.P2 + (t**3)*self.x1
    
    def gradient(self, t: torch.tensor) -> torch.tensor:
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        return (3 * (1 - t)**2 * (self.P1 - self.x0) + 
                6 * (1 - t) * t * (self.P2 - self.P1) + 
                3 * t**2 * (self.x1 - self.P2))
    
    def curve_length(self, t: torch.tensor) -> torch.tensor:
        """Calculate the length of the curve at parameter t
        Args:
            t (torch.tensor): parameter t
        Returns:
            torch.tensor: length of the curve at t
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        return torch.trapz(self.gradient(t).norm(dim=1), t.squeeze(1))


    def forward(self, t: torch.tensor) -> torch.tensor:
        return self.evaluate(t)
    




class FisherRao(nn.Module):
    def __init__(self, curve: nn.Module, decoder: nn.Module, n: int = 1024, batch_size: int = 128):
        super(FisherRao, self).__init__()
        self.curve = curve
        self.n = n #number of equidistant points to use
        self.t = torch.linspace(0,1,n)
        self.dt = self.t[1]
        self.decoder = decoder
        self.batch_size = batch_size

    def energy(self) -> torch.tensor:
        fct = self.decoder(self.curve(self.t)) #Independent(ContinuousBernoulli(logits: torch.Size([256, 1, 28, 28])), 3) object
        probs = fct.base_dist.probs  # Access the logits of the ContinuousBernoulli distribution
        probs_a = probs[:-1]  # All but the last
        probs_b = probs[1:]   # All but the first
        dist_a = td.Independent(td.ContinuousBernoulli(probs=probs_a), 3) # Create a distribution object
        dist_b = td.Independent(td.ContinuousBernoulli(probs=probs_b), 3)
        # Finally, calculate the KL divergence
        return td.kl_divergence(dist_a, dist_b).sum()  # sum over batch

    def optimize_curve(self, n_iter: int = 1000, lr: float = 1e-1, gamma: float = 0.8):
        optimizer = torch.optim.LBFGS(self.curve.parameters(), line_search_fn='strong_wolfe')
        #optimizer = torch.optim.Adam(self.curve.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        tolerance = 1e-1  # Tolerance for parameter change
        previous_params = {name: p.clone() for name, p in self.curve.named_parameters()}
        print('Optimizing curve in observation space')
        print('Initial energy:', self.energy().item())
        #print('Initial curve:', self.curve.P1, self.curve.P2)
        for i in range(n_iter):
            def closure():
                optimizer.zero_grad()  # Clear gradients w.r.t. parameters
                loss = self.energy()  # Compute Loss
                loss.backward()  # Get gradients w.r.t. parameters
                return loss
            optimizer.step(closure)
            scheduler.step()

            # Compute the max parameter change
            max_param_change = max((p - previous_params[name]).abs().max().item() for name, p in self.curve.named_parameters())

            # update previous_params
            previous_params = {name: p.clone() for name, p in self.curve.named_parameters()}
            loss = self.energy()
            print(f'Epoch {i+1}/{n_iter} loss: {loss.item()}')
            #print('Curve:', self.curve.P1, self.curve.P2)
            if max_param_change < tolerance:
                print(f'Training stops at epoch {i} due to small parameter changes.')
                break

    def forward(self) -> float:
        return self.energy()
        

class FisherRaoEnsemble(FisherRao):
    def __init__(self, curve: nn.Module, decoder_list: nn.Module, n: int = 1024, batch_size: int = 128):
        super(FisherRao, self).__init__()
        self.curve = curve
        self.n = n #number of equidistant points to use
        self.t = torch.linspace(0,1,n)
        self.dt = self.t[1]
        self.decoder = decoder_list
        self.batch_size = batch_size
        self.mc_samples = 1 if len(decoder_list) == 1 else 5
        print(f'Using {self.mc_samples} monte carlo samples')

    def energy(self) -> torch.tensor:
        fct_list = [decoder(self.curve(self.t)) for decoder in self.decoder]
        probs_all = torch.stack([fct.base_dist.probs for fct in fct_list])
        monte_carlo_samples = torch.randint(0, len(fct_list), (self.n, self.mc_samples, 2))
        probs_a = []
        probs_b = []
        for i in range(self.n-1):
            probs_a.append(probs_all[monte_carlo_samples[i,:, 0], i])
            probs_b.append(probs_all[monte_carlo_samples[i,:, 1], i+1])
        probs_a = torch.stack(probs_a)
        probs_b = torch.stack(probs_b)
        dist_a = td.Independent(td.ContinuousBernoulli(probs=probs_a), 3)
        dist_b = td.Independent(td.ContinuousBernoulli(probs=probs_b), 3)
        # Calculate KL divergence in a batched way and sum across the last dimension if necessary
        return td.kl_divergence(dist_a, dist_b).sum()/self.mc_samples  # sum over batch


class AbstractDensityMetricGeodesic(nn.Module):
    def __init__(self, curve: nn.Module, data: torch.tensor, n: int = 1024, batch_size: int = 128):
        super(AbstractDensityMetricGeodesic, self).__init__()
        self.curve = curve
        self.data = data
        self.N = data.shape[0] #number of data points
        self.n = n #number of equidistant points to use
        self.t = torch.linspace(0,1,n)
        self.dt = self.t[1]
        self.batch_size = batch_size
        self.sigma = 1
        self.eps = 1e-4


    def evaluate_p_x_batch(self, x: torch.tensor) -> torch.tensor:
        """evaluate the density of a batch of points x under the Gaussian kernel density estimator"""
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        M, D = x.shape  # Number of points and dimensionality
        normal_dists = td.Normal(self.data.unsqueeze(1).expand(-1, M, -1), self.sigma)
        pdf_values = normal_dists.log_prob(x.expand(self.N, -1, -1)).exp()
        pdf_values = pdf_values.prod(dim=2).mean(dim=0) 
        return pdf_values
    
    def riemannian_metric(self, x: torch.tensor) -> torch.tensor:
        px = self.evaluate_p_x_batch(x)
        v = 1/(px+1e-4)
        n = x.shape[1]
        G = torch.zeros((x.shape[0], n, n))
        indices = torch.arange(n)
        G[:, indices, indices] = v[:,None]
        return G

    def energy(self) -> torch.tensor:
        curve_points = self.curve(self.t)
        curve_grad = self.curve.gradient(self.t).unsqueeze(-1) # shape -> [n, d, 1]
        G = self.riemannian_metric(curve_points) #shape -> [n, d, d]
        Gx = torch.matmul(G, curve_grad)  # Shape: [n, d, 1]
        curve_grad = curve_grad.transpose(1, 2)  # Shape: [n, 1, d]
        integrand = torch.matmul(curve_grad, Gx).squeeze(-1).squeeze(-1)  # Shape: [1024]
        return torch.trapz(integrand , self.t)
    
    def forward(self) -> torch.tensor:
        return self.energy()

    def optimize_curve(self, n_iter: int = 1000, lr: float = 1e-1, gamma: float = 0.8):
        optimizer = torch.optim.LBFGS(self.curve.parameters(), line_search_fn='strong_wolfe')
        # optimizer = torch.optim.Adam(self.curve.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        tolerance = 1e-2  # Tolerance for parameter change
        previous_params = {name: p.clone() for name, p in self.curve.named_parameters()}
        print('Optimizing curve in observation space')
        print('Initial energy:', self.energy().item())
        #print('Initial curve:', self.curve.P1, self.curve.P2)
        for i in range(n_iter):
            def closure():
                optimizer.zero_grad()  # Clear gradients w.r.t. parameters
                loss = self.energy()  # Compute Loss
                loss.backward()  # Get gradients w.r.t. parameters
                return loss
            optimizer.step(closure)
            scheduler.step()

            # Compute the max parameter change
            max_param_change = max((p - previous_params[name]).abs().max().item() for name, p in self.curve.named_parameters())

            # update previous_params
            previous_params = {name: p.clone() for name, p in self.curve.named_parameters()}
            loss = self.energy()
            print(f'Epoch {i+1}/{n_iter} loss: {loss.item()}')
            #print('Curve:', self.curve.P1, self.curve.P2)
            if max_param_change < tolerance:
                print(f'Training stops at epoch {i} due to small parameter changes.')
                break



def proximity(curve_points, latent):
    """
    Compute the average distance between points on a curve and a collection
    of latent variables.

    Parameters:
    curve_points: [torch.tensor]
        M points along a curve in latent space. tensor shape: M x latent_dim
    latent: [torch.tensor]
        N points in latent space (latent means). tensor shape: N x latent_dim

    The function returns a scalar.
    """
    pd = torch.cdist(curve_points, latent)  # M x N
    pd_min, _ = torch.min(pd, dim=1)
    pd_min_max = pd_min.max()
    return pd_min_max