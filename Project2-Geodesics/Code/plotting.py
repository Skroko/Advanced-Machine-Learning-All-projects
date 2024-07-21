import torch
import numpy as np
import matplotlib.pyplot as plt
XLIM = (-7, 5.5)
YLIM = (-5.7, 5.7)
VMIN = -1329.2173
VMAX = -1162.3278
# VMIN = 0
# VMAX = None
def plot_entropy(ax, decoder, grid_size=(100,101)):
    """
    Plot the entropy of the decoder as a function of the latent space.
    """
    #get x and y limits
    # x = torch.linspace(*ax.get_xlim(), grid_size[0])
    # y = torch.linspace(*ax.get_ylim(), grid_size[1])
    x = torch.linspace(*XLIM, grid_size[0])
    y = torch.linspace(*YLIM, grid_size[1])
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.stack((X,Y), dim=-1).reshape(-1,2)
    #if decoder is nn.ModuleList, average over all decoders
    if isinstance(decoder, torch.nn.ModuleList):
        entropy = 0
        for d in decoder:
            entropy += d(Z).entropy().reshape(grid_size).detach().numpy()
            #entropy += d(Z).variance.mean(dim=(1,2,3)).reshape(grid_size).detach().numpy()
        entropy /= len(decoder)
    else:
        entropy = decoder(Z).entropy().reshape(grid_size).detach().numpy()
        #print(decoder(Z).variance.mean(dim=(1,2,3)).shape)
        #entropy = decoder(Z).variance.mean(dim=(1,2,3)).reshape(grid_size).detach().numpy()
    im= ax.imshow(entropy.T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='hot', zorder=0, interpolation='bilinear', alpha=0.5, vmin=VMIN, vmax=VMAX)
    #ax.contour(X,Y,entropy, cmap='hot', zorder=0, levels=20)
    #print(entropy.min(), entropy.max())
    return im

def plot_data_density(ax, density_func, grid_size=(100,101)):
    """
    Plot the density of the data in the latent space.
    """
    #get x and y limits
    # x = torch.linspace(*ax.get_xlim(), grid_size[0])
    # y = torch.linspace(*ax.get_ylim(), grid_size[1])
    x = torch.linspace(*XLIM, grid_size[0])
    y = torch.linspace(*YLIM, grid_size[1])
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.stack((X,Y), dim=-1).reshape(-1,2)
    density = density_func(Z).reshape(grid_size).detach().numpy()
    ax.imshow(density.T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower',alpha=0.5, cmap='hot', zorder=0, interpolation='bilinear')
    ax.contour(X,Y,density, zorder=0, alpha=0.5, levels=20)

def plot_proximities(proximities, save_path):
    """
    Plot the proximities of the ensemble members.
    """
    # kep rows where the row sum is larger than 0.0
    # kep rows there row sum is larger than 0.0
    proximities = proximities.detach().numpy()
    proximities = proximities[proximities.min(axis=1) > 0.0]
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    plt.plot(range(1, 11), proximities.mean(axis=0))
    ax.set_xlabel('Number of ensemble members')
    ax.set_ylabel('Proximity')
    ax.set_xticks(range(1,11))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
   