#%%
import torch
import matplotlib.pyplot as plt
from vae import GraphLevelVAE
from trainer import GNNTrainer
from dataloader import GraphDataLoader


seed = 0
device = 'cuda'
dataset_name = 'MUTAG'
# (train, validation, test) must sum to the total number of graphs in the dataset (188)
# we do not need a test set for this assignment as we implement a generative model
batch_size = (144, 44, 0) 

#set up data loader
dataloader = GraphDataLoader(dataset_name, batch_size=batch_size, device=device, seed=seed)


#set up model
model = GraphLevelVAE().to(device)


#  Train the model
model.train()
GNNTrainer(model, dataloader, num_epochs=2000,lr=1., gamma=0.9975).train()

#save the model
torch.save(model.state_dict(), 'model.pth')




latents = []
for data in dataloader.train_loader:
    q = model.encoder(data.x, data.edge_index, data.batch)
    latents.append(q.rsample())
latents = torch.cat(latents).detach().numpy()
plt.scatter(latents[:, 0], latents[:, 1], c='b')
latents = []
for data in dataloader.validation_loader:
    q = model.encoder(data.x, data.edge_index, data.batch)
    latents.append(q.rsample())
latents = torch.cat(latents).detach().numpy()
plt.scatter(latents[:, 0], latents[:, 1], c='r')
plt.show()
dataloader.plot_graphs_from_adjacency_matrix(model.sample(10))
# %%
