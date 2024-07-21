#%%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
import matplotlib.pyplot as plt


#%%
class GNNTrainer():
    def __init__(self, model, dataloader, num_epochs=500, lr=1., gamma=0.995):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        self.device = dataloader.device
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.train_loader = dataloader.train_loader
        self.validation_loader = dataloader.validation_loader

        

    def train(self):
        self.train_losses = []
        self.validation_losses = []


        #plt.ion()

        self.model.train()

        for epoch in range(self.num_epochs):
            self.model.train()
            self.train_loss = 0.
            for data in self.train_loader:
                self.train_step(data)

            with torch.no_grad():    
                self.model.eval()
                self.validation_loss = 0.
                for data in self.validation_loader:
                    self.validation_step(data)
                
                self.store_metrics(epoch)
                

    def train_step(self, data):
        loss = self.model(data.x, data.edge_index, batch=data.batch)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.train_loss += loss.detach().cpu().item() * data.batch_size / self.dataloader.n_train

        return loss
    
    def validation_step(self, data):
        loss = self.model(data.x, data.edge_index, data.batch)
        self.validation_loss += loss.detach().cpu().item() * data.batch_size / self.dataloader.n_val
        return loss
    
    def store_metrics(self, epoch):
        # Store the training and validation accuracy and loss for plotting
        self.train_losses.append(self.train_loss)
        self.validation_losses.append(self.validation_loss)

        # Print stats and update plots
        if (epoch+1)%50 == 0:
            print(f'Epoch {epoch+1}')
            print(f'- Learning rate   = {self.scheduler.get_last_lr()[0]:.1e}')
            print(f'         loss     = {self.train_loss:.3f}')
            print(f'         loss     = {self.validation_loss:.3f}')
