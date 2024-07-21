# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen and Søren Hauberg, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
import random
import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
import torch.utils.data
import json
from torchvision.utils import save_image
from tqdm import tqdm
from VAE import (
    VAE,
    GaussianPrior,
    GaussianEncoder,
    BernoulliDecoder,
    encoder_net,
    new_decoder,
    VAEEnsemble,
)
from trainer import train
from plotting import plot_entropy, plot_data_density, plot_proximities
from torchvision import datasets, transforms
import glob
import argparse
import os
import numpy as np
from curves import (
    FisherRao,
    ThirdOrderPolynomialCurve,
    FisherRaoEnsemble,
    AbstractDensityMetricGeodesic,
    proximity,
)


if __name__ == "__main__":
    # Parse arguments from command line ##############################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "plot_ensemble_model", "plot_non_ensemble_model"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pt",
        help="file to save model to or load model from (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-ensembles",
        type=int,
        default=10,
        metavar="N",
        help="number of ensembles for the model (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=123133, metavar="N", help="random seed"
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot",
    )
    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    # Set random seed seed for reproducibility ###############################################################################################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device ################################################################################################################################
    device = args.device

    # Load a subset of MNIST and create data loaders ############################################################################################
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]
        return torch.utils.data.TensorDataset(new_data, new_targets)

    # size of the dataset
    num_train_data, num_test_data, num_classes = 2048, 2048, 3
    # Load train data
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    # Load test data
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_test_data, num_classes
    )
    mnist_test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Define VAE model ########################################################################################################################
    M = args.latent_dim  # latent dimension
    prior, encoder = GaussianPrior(M), GaussianEncoder(
        encoder_net(M)
    )  # gaussian prior and encoder

    # Define either single decoder or ensemble of decoders
    num_ensembles = args.num_ensembles
    if num_ensembles > 1:
        print("Using ensemble of decoders")
        decoder_list = nn.ModuleList(
            [BernoulliDecoder(new_decoder(M)) for _ in range(num_ensembles)]
        )
        model = VAEEnsemble(prior, decoder_list, encoder).to(device)
    else:
        print("Using single decoder")
        decoder = BernoulliDecoder(new_decoder(M))
        model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run #######################################################################################################################
    ###########################################################################################################################################
    ###########################################################################################################################################
    if args.mode == "train":
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Train model
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs * num_ensembles,
            args.device,
        )  # multiply epochs by number of ensembles
        # Save model
        output_dir = f"./models/{args.model}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

        # plot samples from the model as a sanity check for reasonable output
        with torch.no_grad():
            if num_ensembles > 1:
                for i in range(num_ensembles):
                    model.set_decoder(i)
                    samples = (model.sample(64)).cpu()
                    save_image(
                        samples.view(64, 1, 28, 28),
                        os.path.join(output_dir, f"sample_{i}.png"),
                        nrow=8,
                    )
            else:
                samples = (model.sample(64)).cpu()
                save_image(
                    samples.view(64, 1, 28, 28),
                    os.path.join(output_dir, f"sample.png"),
                    nrow=8,
                )

    ###########################################################################################################################################
    elif args.mode == "plot_ensemble_model":
        import pickle
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        plt.style.use("seaborn-v0_8-whitegrid")

        ## Load trained model
        output_dir = f"./models/{args.model}"
        model_path = os.path.join(output_dir, "model.pt")
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(args.device))
        )
        model.eval()
        model.set_decoder(0)

        ## Encode test and train data
        latents, labels = [], []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                z = model.encoder((x).float())
                latents.append(z.mean)
                labels.append(y)
            latents = torch.concatenate(latents, dim=0)
            labels = torch.concatenate(labels, dim=0)

        ## Initialize plot
        fig, ax = plt.subplots(2, 2, figsize=(10, 9.15), sharex=True, sharey=True)
        [a.grid(False) for a in ax.flatten()]
        for a in ax.flatten():
            a.tick_params(axis="both", which="major", labelsize=14)
        annotate_plot = lambda ax, text: ax.text(
            0.06,
            0.95,
            text,
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            color="white",
            zorder=1,
        )
        [
            annotate_plot(a, text)
            for a, text in zip(ax.flatten(), ["a)", "b)", "c)", "d)"])
        ]

        def update_plot(fig):
            fig.subplots_adjust(wspace=0.0, hspace=0.0)
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, 'big_plot.png'), dpi=300, bbox_inches="tight"
            )

        for k in range(num_classes):
            idx = labels == k
            plt.scatter(latents[idx, 1].numpy(force=True), latents[idx, 0].numpy(force=True), label=f'Class {k}', alpha=0.7)
            ax[0,1].scatter(latents[idx, 1].numpy(force=True), latents[idx, 0].numpy(force=True), label=f'Class {k}', alpha=0.7)
            ax[0,0].scatter(latents[idx, 1].numpy(force=True), latents[idx, 0].numpy(force=True), label=f'Class {k}', alpha=0.7)
            ax[1,0].scatter(latents[idx, 1].numpy(force=True), latents[idx, 0].numpy(force=True), label=f'Class {k}', alpha=0.7)
        plt.legend()
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()

        #for k in range(num_classes):
        #    idx = labels == k
        #    colors = ["r", "g", "b"]
        #    [
        #        a.scatter(
        #            latents[idx, 0],
        #            latents[idx, 1],
        #            label=f"{k}",
        #            alpha=0.8,
        #            s=10,
        #            c=colors[k],
        #            edgecolors="k",
        #        )
        #        for a in ax.flatten()
        #    ]

        ## Plot entropy
        #plot_entropy(ax[1, 1], model.decoder, grid_size=(100, 101))
        #im = plot_entropy(ax[1, 0], model.decoders, grid_size=(100, 101))
        ## add common colorbar
        #cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03])
        #cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
        #cbar.set_label("Avg. Entropy", labelpad=15, fontsize=14)
        #cbar.ax.tick_params(labelsize=14)

        ## Plot data density
        ADM = AbstractDensityMetricGeodesic(
            None, latents
        )  # initialize the abstract density metric without a curve in order to get access to the density function
        #plot_data_density(ax[0, 0], ADM.evaluate_p_x_batch, grid_size=(100, 101))
        #plot_data_density(ax[0, 1], ADM.evaluate_p_x_batch, grid_size=(100, 101))

        ## Plot random geodesics
        num_curves = args.num_curves  # number of geodesics to plot
        num_subintervals = 128  # number of subintervals to use for the geodesic
        n_iter = 50  # max number of iterations to optimize the geodesic
        # load the indices of the curves to plot
        with open("indices.json", "r") as f:
            curve_indices = torch.tensor(json.load(f))
        curve_indices = curve_indices[:num_curves]

        # initialize the curve parameter
        t = torch.linspace(0, 1, 100)[:, None]  # curve parameter
        prox = torch.zeros(num_curves, num_ensembles)

        # delete csv files if they exist
        if os.path.exists(os.path.join(output_dir, "euclidean_distances.csv")):
            os.remove(os.path.join(output_dir, "euclidean_distances.csv"))
        if os.path.exists(
            os.path.join(output_dir, "curve_energy_no_ensembles.csv.csv")
        ):
            os.remove(os.path.join(output_dir, "curve_energy_no_ensembles.csv.csv"))
        if os.path.exists(os.path.join(output_dir, "curve_energy_ensembles.csv.csv")):
            os.remove(os.path.join(output_dir, "curve_energy_ensembles.csv.csv"))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        picked_colors = [ random.choice(colors) for _ in range(num_curves)]
        import random
        # add a curve to the plot
        def plot_curve(ax, curve, idx):
            z = curve(t).detach().numpy()
            ax.plot(z[:, 1], z[:, 0], color=picked_colors[idx], linestyle="-")

        # Loop over the curves
        for k in tqdm(range(num_curves)):
            #random color from a palette of 10 colors
            color = plt.cm.tab10(k)

            i, j = curve_indices[k, 0], curve_indices[k, 1]
            z0, z1 = latents[i], latents[j]

            # add euclidean distance to a csv file
            with open(os.path.join(output_dir, "euclidean_distances.csv"), "a") as f:
                f.write(f"{i},{j},{torch.norm(z0-z1).item()}\n")

            # initialize the curve
            curve = ThirdOrderPolynomialCurve(z0, z1)
            plot_curve(ax[0, 0], curve, idx=k)

            # initialize by optimizing the curve under the Abstract Density Metric
            ADM = AbstractDensityMetricGeodesic(
                curve, latents, n=num_subintervals, batch_size=128
            )
            ADM.optimize_curve(n_iter=n_iter, lr=0.1, gamma=1)
            plot_curve(ax[0, 1], curve, idx=k)

            # make a copy of the initializated curve
            curve1 = ThirdOrderPolynomialCurve(z0, z1)
            curves = []
            # If there are multiple decoders, optimize the curve under the Fisher-Rao metric using an varying number of decoders
            for e in range(0, model.num_ensembles):
                if e!=0 and e!=9:
                    continue
                # reset the curve to abstract density metric geodesic
                curve1.set_control_points(curve.P1.data.clone(), curve.P2.data.clone())
                selected_decoders = model.decoders[: e + 1]
                distance_measure = FisherRaoEnsemble(
                    curve1, selected_decoders, n=num_subintervals, batch_size=128
                )
                print(torch.norm(z0-z1).item()/10)
                distance_measure.optimize_curve(n_iter=n_iter, lr=0.1, gamma=1)
                # calculate the proximities
                curve_points = curve1(t)
                prox[k, e] = proximity(curve_points, latents)
                plot_proximities(
                    prox, os.path.join(output_dir, "proximities.pdf")
                )  # update the plot of proximities
                # save the proximities to a pickle file
                with open(os.path.join(output_dir, "proximities.pkl"), "wb") as f:
                    pickle.dump(prox, f)
                if e == 0:
                    plot_curve(ax[1, 0], curve1, idx=k)
                    with open(
                        os.path.join(output_dir, "curve_energy_no_ensembles.csv"), "a"
                    ) as f:
                        f.write(f"{i},{j},{distance_measure().item()}\n")
                elif e == 9:
                    plot_curve(ax[1, 1], curve1, idx=k)
                    with open(
                        os.path.join(output_dir, "curve_energy_ensembles.csv"), "a"
                    ) as f:
                        f.write(f"{i},{j},{distance_measure().item()}\n")
                    
                    curves.append(curve1(t).detach().numpy())
                    np.save(os.path.join(output_dir, "curves_10ensembles.npy"), curves)
            update_plot(fig)
        
        ###########################################################################################################################################
    elif args.mode == "plot_non_ensemble_model":
        import pickle
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        plt.style.use("seaborn-v0_8-whitegrid")

        ## Load trained model
        output_dir = f"./models/{args.model}"
        model_path = os.path.join(output_dir, "model.pt")
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(args.device))
        )
        model.eval()

        ## Encode test and train data
        latents, labels = [], []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                z = model.encoder(x)
                latents.append(z.mean)
                labels.append(y)
            latents = torch.concatenate(latents, dim=0)
            labels = torch.concatenate(labels, dim=0)

        ## Initialize plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 9.15/2), sharex=True, sharey=True)
        [a.grid(False) for a in ax.flatten()]
        for a in ax.flatten():
            a.tick_params(axis="both", which="major", labelsize=14)
        annotate_plot = lambda ax, text: ax.text(
            0.06,
            0.95,
            text,
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            color="white",
            zorder=1,
        )
        [
            annotate_plot(a, text)
            for a, text in zip(ax.flatten(), ["a)", "b)"])
        ]

        def update_plot(fig):
            fig.subplots_adjust(wspace=0.0, hspace=0.0)
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, 'big_plot.png'), dpi=300, bbox_inches="tight"
            )

        for k in range(num_classes):
            idx = labels == k
            colors = ["r", "g", "b"]
            [
                a.scatter(
                    latents[idx, 0],
                    latents[idx, 1],
                    label=f"{k}",
                    alpha=0.8,
                    s=10,
                    c=colors[k],
                    edgecolors="k",
                )
                for a in ax.flatten()
            ]

        ## Plot entropy
        plot_entropy(ax[0], model.decoder, grid_size=(100, 101))
        im = plot_entropy(ax[1], model.decoder, grid_size=(100, 101))
        # add common colorbar
        cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Avg. Entropy", labelpad=15, fontsize=14)
        cbar.ax.tick_params(labelsize=14)


        ## Plot random geodesics
        num_curves = args.num_curves  # number of geodesics to plot
        num_subintervals = 128  # number of subintervals to use for the geodesic
        n_iter = 50  # max number of iterations to optimize the geodesic
        # load the indices of the curves to plot
        with open("indices.json", "r") as f:
            curve_indices = torch.tensor(json.load(f))
        curve_indices = curve_indices[:num_curves]

        # initialize the curve parameter
        t = torch.linspace(0, 1, 100)[:, None]  # curve parameter
        prox = torch.zeros(num_curves, num_ensembles)

        # delete csv files if they exist
        if os.path.exists(os.path.join(output_dir, "euclidean_distances.csv")):
            os.remove(os.path.join(output_dir, "euclidean_distances.csv"))
        if os.path.exists(
            os.path.join(output_dir, "curve_energy_no_ensembles.csv.csv")
        ):
            os.remove(os.path.join(output_dir, "curve_energy_no_ensembles.csv.csv"))
        if os.path.exists(
            os.path.join(output_dir, "curve_energy_linear_initialisation.csv")
        ):
            os.remove(os.path.join(output_dir, "curve_energy_linear_initialisation.csv"))
        if os.path.exists(
            os.path.join(output_dir, "curve_energy_ADM_initialisation.csv")
        ):
            os.remove(os.path.join(output_dir, "curve_energy_ADM_initialisation.csv"))

        # add a curve to the plot
        def plot_curve(ax, curve, color="orange"):
            z = curve(t).detach().numpy()
            ax.plot(z[:, 0], z[:, 1], color=color, linestyle="-")

        # Loop over the curves
        curves = []
        for k in tqdm(range(num_curves)):
            i, j = curve_indices[k, 0], curve_indices[k, 1]
            z0, z1 = latents[i], latents[j]

            # add euclidean distance to a csv file
            with open(os.path.join(output_dir, "euclidean_distances.csv"), "a") as f:
                f.write(f"{i},{j},{torch.norm(z0-z1).item()}\n")

            # initialize the curve
            curve = ThirdOrderPolynomialCurve(z0, z1)
            plot_curve(ax[0], curve, color="orange")

            # initialize by optimizing the curve under the Abstract Density Metric
            # ADM = AbstractDensityMetricGeodesic(
            #     curve, latents, n=num_subintervals, batch_size=128
            # )
            # with open(
            #     os.path.join(output_dir, "curve_energy_linear_initialisation.csv"), "a"
            # ) as f:
            #     f.write(f"{i},{j},{ADM().item()}\n")
            # ADM.optimize_curve(n_iter=n_iter, lr=1, gamma=0.8)
            plot_curve(ax[1], curve, color="orange")
            # with open(
            #     os.path.join(output_dir, "curve_energy_ADM_initialisation.csv"), "a"
            # ) as f:
            #     f.write(f"{i},{j},{ADM().item()}\n")

            # make a copy of the initializated curve
            curve1 = ThirdOrderPolynomialCurve(z0, z1)
            curve1.set_control_points(curve.P1.data.clone(), curve.P2.data.clone())
            distance_measure = FisherRao(
                    curve1, model.decoder, n=num_subintervals, batch_size=128
                )
            distance_measure.optimize_curve(n_iter=n_iter, lr=1, gamma=0.8)
            plot_curve(ax[2], curve1, color="orange")
            with open(
                os.path.join(output_dir, "curve_energy_no_ensembles.csv"), "a"
            ) as f:
                f.write(f"{i},{j},{distance_measure().item()}\n")
            update_plot(fig)

            curves = [curve1(t).detach().numpy() for curve1 in curves]
            np.save(os.path.join(output_dir, "curves.npy"), curves)