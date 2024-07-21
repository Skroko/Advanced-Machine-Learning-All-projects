import numpy as np
import matplotlib.pyplot as plt
import torch




def plot_vae_samples_and_contours(vae, data_loader,grid_size=300, device='cpu'):
    """
    Plot samples from the aggregated posterior and prior contours.

    Parameters:
    vae: [VAE] 
        Trained instance of the VAE model.
    data_dataloader: data_loader
    n_samples: [int] 
        Number of samples to draw from the prior and posterior.
    grid_size: [int] 
        The size of the grid for plotting the prior contours.
    """
    vae.eval()

    # Sample from the aggregated posterior
    posterior_samples = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = vae.encoder(x)
            posterior_samples.append(q.sample().squeeze().cpu().detach().numpy())
            labels.append(y.numpy())
    posterior_samples = np.concatenate(posterior_samples, axis=0)
    labels = np.concatenate(labels, axis=0)

    #include 100 random samples from the posterior
    # idx = np.random.choice(posterior_samples.shape[0], 1000, replace=False)
    # posterior_samples = posterior_samples[idx]
    # labels = labels[idx]

    # Perform PCA if the latent space is high-dimensional
    if posterior_samples.shape[1] > 2:
        print("The latent space is too high-dimensional to plot, so performing PCA")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        posterior_samples = pca.fit_transform(posterior_samples)

    # Generate a grid of values for plotting the contours
    x = np.linspace(posterior_samples[:, 0].min(), 
                    posterior_samples[:, 0].max(), grid_size)
    y = np.linspace( posterior_samples[:, 1].min(), 
                    posterior_samples[:, 1].max(), grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Compute the density of the prior at each point in the grid
    prior_density = np.exp(vae.prior().log_prob(torch.tensor(pos, dtype=torch.float)).cpu().detach().numpy())

    # Plot the contours of the prior
    #plt.contourf(X, Y, prior_density, levels=100, cmap='viridis')
    prior_density[prior_density > 0.01]=0.01
    cp=plt.contourf(X, Y, prior_density, levels=np.arange(0,0.0011,0.0001))
    plt.colorbar(cp)

    # Plot the aggregated posterior samples
    im=plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.4, label='Aggregated Posterior Samples', s=1, c=labels, cmap='tab10')
    im.set_clim(-0.5, 9.5)
    cbar = plt.gcf().colorbar(im, ax=plt.gca(),ticks=np.arange(0, 10, 1.0), orientation='horizontal')
    cbar.solids.set(alpha=1)

    plt.title(f'Prior Contours and Aggregated Posterior Samples.')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.tight_layout()
    # plt.legend()




