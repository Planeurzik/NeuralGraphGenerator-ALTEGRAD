import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch_geometric.loader import DataLoader

from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset

import csv

from datetime import datetime

nb_epochs = 50

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj

trainset = preprocess_dataset("train", 50, 10)
validset = preprocess_dataset("valid", 50, 10)
testset = preprocess_dataset("test", 50, 10)

train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
val_loader = DataLoader(validset, batch_size=256, shuffle=False)
test_loader = DataLoader(testset, batch_size=256, shuffle=False)

class BasicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(BasicEncoder, self).__init__()
        self.fc1 = nn.Linear(8, hidden_dim) # From 8-feature vector to hidden_dim to adapt it
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Mandatory, else not converge
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        data = x.stats
        data = F.relu(self.bn1(self.fc1(data)))
        out = self.fc2(data)
        return out

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = BasicEncoder(input_dim, hidden_dim_enc, hidden_dim_enc)#, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        # Check if adj is symmetrical on the last two dimensions, useless, always symmetrical
        # if not torch.allclose(adj, adj.transpose(-1, -2), atol=1e-5):
        #    print("Warning: adj is not symmetrical")
        
        recon = F.binary_cross_entropy(adj, data.A, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld
    
# initialize VGAE model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = VariationalAutoEncoder(11, 16, 256, 32, 2, 3, 50).to(device)

#autoencoder_model_train_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
#print("Autoencoder params:", autoencoder_model_train_params)
#exit()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

best_val_loss = np.inf
for epoch in range(1, nb_epochs+1):
    autoencoder.train()
    train_loss_all = 0
    train_count = 0
    train_loss_all_recon = 0
    train_loss_all_kld = 0
    cnt_train=0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss, recon, kld  = autoencoder.loss_function(data)
        train_loss_all_recon += recon.item()
        train_loss_all_kld += kld.item()
        cnt_train+=1
        loss.backward()
        train_loss_all += loss.item()
        train_count += torch.max(data.batch)+1
        optimizer.step()

    autoencoder.eval()
    val_loss_all = 0
    val_count = 0
    cnt_val = 0
    val_loss_all_recon = 0
    val_loss_all_kld = 0

    for data in val_loader:
        data = data.to(device)
        loss, recon, kld  = autoencoder.loss_function(data)
        val_loss_all_recon += recon.item()
        val_loss_all_kld += kld.item()
        val_loss_all += loss.item()
        cnt_val+=1
        val_count += torch.max(data.batch)+1

    if epoch % 1 == 0:
        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val))
        
    scheduler.step()

    if best_val_loss >= val_loss_all:
        best_val_loss = val_loss_all
        torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder_VAE.pth.tar')

checkpoint = torch.load('autoencoder_VAE.pth.tar')
autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()

with open("test_VAE.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(test_loader):
        data = data.to(device)
        
        stat = data.stats
        
        bs = stat.size(0)

        graph_ids = data.filename
        
        adj = autoencoder(data)


        for i in range(adj.size(0)):

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])