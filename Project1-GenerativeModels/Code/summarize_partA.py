from pathlib import Path
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch


from src.utilities.load_model import load_model
from src.data.dataloader import get_MNIST_dataloader

CSV_FILE = Path("reports/PartA_metrics.csv")


#load csv file as dataframe
df = pd.read_csv(CSV_FILE)

res_dict ={}
for index, row in df.iterrows():
    model_name =Path(row.model_dir).parent.name
    if model_name not in res_dict:
        res_dict[model_name] = []
    res_dict[model_name].append((row.ELBO, row.IWAE))

#build now a df with the mean and std of the metrics
res = []
for model, metrics in res_dict.items():
    elbo = np.array([x[0] for x in metrics])
    iwae = np.array([x[1] for x in metrics])
    res.append((model, elbo.mean(), elbo.std(), iwae.mean(), iwae.std()))

res_df = pd.DataFrame(res, columns=["model", "ELBO_mean", "ELBO_std", "IWAE_mean", "IWAE_std"])

#save the new df
res_df.to_csv(Path("reports/PartA_metrics_aggregated.csv"), index=False)

# plot the results


def plot_vae_samples_and_contours(vae, data_loader,grid_size=300, device='cpu',levels=None,vmax=None):
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
    print(prior_density.max())

    # Plot the contours of the prior
    #plt.contourf(X, Y, prior_density, levels=100, cmap='viridis')


    #cp=plt.contourf(X, Y, prior_density, levels=levels)
    plt.imshow(prior_density, extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis', origin='lower',vmax=vmax)
    #make aspect to fit the plot
    plt.gca().set_aspect('auto', adjustable='box')
    # plt.colorbar(cp)

    # Plot the aggregated posterior samples
    im=plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.4, label='Aggregated Posterior Samples', s=1, c=labels, cmap='tab10')
    # im.set_clim(-0.5, 9.5)
    # cbar = plt.gcf().colorbar(im, ax=plt.gca(),ticks=np.arange(0, 10, 1.0), orientation='horizontal')
    # cbar.solids.set(alpha=1)

    plt.tight_layout()
    # plt.legend()


model_dirs = [Path('models/partA_Gaussian_v1/1708045601'),
              Path('models/partA_gmm_v1/1708039668'),
              Path('models/partA_vampPrior_v1/1708043636'),
              Path('models/partA_flow_v1/1708039015')]
titles = ['Gaussian Prior', 'MoG Prior', 'VampPrior', 'Flow Prior']
vmax = [0.05,0.005,0.001,0.005]
models = [load_model(model_dir) for model_dir in model_dirs]

_, test_loader = get_MNIST_dataloader(batch_size=1000, transform_description='binarized')




fig = plt.subplots(2,2, figsize=(8,8))
for i, model in enumerate(models):
    plt.subplot(2,2,i+1)
    plot_vae_samples_and_contours(model, test_loader, device='cpu', vmax=vmax[i])
    plt.title(titles[i], fontsize=12)
    if i in [2,3]:
        plt.xlabel(r"$z_1$", fontsize=12)
    if i in [0,2]:
        plt.ylabel(r"$z_2$", fontsize=12)
plt.tight_layout()
plt.savefig(Path("reports/PartA_latent_space.png"), bbox_inches="tight", dpi=300)


