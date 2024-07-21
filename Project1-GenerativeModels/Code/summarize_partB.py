from pathlib import Path
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


from AMLsrc.utilities.load_model import load_model, load_flow_model
from AMLsrc.data.dataloader import get_MNIST_dataloader

#set seed
print("Setting seed")
torch.manual_seed(1236)
batch_size=32
_, mnist_test_loader = get_MNIST_dataloader(batch_size=batch_size, transform_description='standard')


model = ['',load_model(Path('models/partB_VAE_v1/1708260893')),
       #load_flow_model(Path('models/partB_flow_test_version/1708116431')),
        load_flow_model(Path('models/partB_flow_v1/1708640122')),
       load_flow_model(Path('models/partB_ddpm_v1/1708259498'))
        ]

titles = ['MNIST', 'VAE', 'Flow', 'DDMP']

fig, ax = plt.subplots(4, 8, figsize=(10, 5))
for i in range(len(model)):
    print(i)
    for j in range(8):
        if i == 0:
            sample = mnist_test_loader.dataset[j][0].view(28, 28)
        else:
            with torch.no_grad():
                if i == 3:
                    sample = model[i].sample((1,784)).view(28, 28)
                    # convert to 0-1 range
                    sample = (sample + 1) / 2
                else:
                    sample = model[i].sample(1).view(28, 28)
                #crop to 0-1 range
                sample = torch.clamp(sample, 0, 1)
        print(sample.min(), sample.max())
        ax[i, j].imshow(sample, cmap='gray')
        if j == 0:
            ax[i, j].set_ylabel(titles[i], fontsize=15)
        #remove ticks and tick labels
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])


plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('reports/PartB_samples.png', bbox_inches='tight', dpi=300)

# compute FID and KID
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

#write header to txt file
with open('reports/PartB_FID_KID.txt', 'w') as f:
    f.write('Model, KID_mean, KID_std, FID\n')

with torch.no_grad():
    #get test_data
    test_data = []
    n=256
    for x, _ in mnist_test_loader:
        test_data.append(x)
        if len(test_data) >=n // batch_size:
            break
    test_data = torch.cat(test_data, 0).unsqueeze(1)[:n].repeat(1,3,1,1)
    print("Test data shape")
    print(test_data.shape)


    models = model[1:]
    for i in range(len(models)):
        if i == 2:
            model_samples = []
            for j in tqdm(range(n)):
                model_samples.append(models[i].sample((1,784)).view(1, 1, 28, 28))
            model_samples = torch.cat(model_samples, 0)
            model_samples = (model_samples + 1) / 2
            model_samples = model_samples.repeat(1,3,1,1)
        else:
            model_samples = models[i].sample(n).view(n, 1, 28, 28)
            model_samples = model_samples.repeat(1,3,1,1)
        #crop to 0-1 range
        model_samples = torch.clamp(model_samples, 0, 1)
        print(f"Calculating KID for {titles[i+1]}")
        print(model_samples.shape)
        kid = KernelInceptionDistance(subset_size=50, normalize=True)
        kid.update(test_data, real=True)
        kid.update(model_samples, real=False)
        mean_kid, std_kid = kid.compute()
        print(f"The mean KID {mean_kid} +- {std_kid}")
        del kid
        print(f"Calculating FID for {titles[i+1]}")
        fid = FrechetInceptionDistance(feature=64, normalize=True)
        fid.update(test_data, real=True)
        fid.update(model_samples, real=False)
        fid = fid.compute()
        print(f"FID {fid}")

        with open('reports/PartB_FID_KID.txt', 'a') as f:
            f.write(f'{titles[i+1]}, {mean_kid}, {std_kid}, {fid}\n')
