import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

#from extract_feats import extract_feats, extract_numbers

from new_extract_feats import get_sentence_features, extract_feats



def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):

    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_'+dataset+'.pt'
        desc_file = './data/'+dataset+'/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = get_sentence_features(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename = graph_id))
            fr.close()                    
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = './data/dataset_'+dataset+'.pt'
        graph_path = './data/'+dataset+'/graph'
        desc_path = './data/'+dataset+'/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path,fileread)
                fstats = os.path.join(desc_path,filen+".txt")
                #load dataset to networkx
                if extension=="graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:,idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)
                mn = min(G.number_of_nodes(),spectral_emb_dim)
                mn+=1
                x[:,1:mn] = eigvecs[:,:spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename = filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst


        

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start




# Graph augmentation
import random
import copy
from torch_geometric.utils import dropout_edge, subgraph

def graph_augment(data, drop_node_prob=0.0, drop_edge_prob=0.1, mask_feature_prob=0.1):
    """
    Creates a randomly augmented version of 'data' by:
      1) Dropping a fraction of nodes (and their edges).
      2) Dropping a fraction of edges.
      3) Masking a fraction of node features.
    
    Args:
        data (torch_geometric.data.Data): The original graph data object.
        drop_node_prob (float): Probability of dropping each node. Range [0, 1].
        drop_edge_prob (float): Probability of dropping each edge. Range [0, 1].
        mask_feature_prob (float): Probability of masking each node-feature entry. Range [0, 1].

    Returns:
        data_aug (torch_geometric.data.Data): Augmented graph data.
    """
    data_aug = copy.deepcopy(data)

    # -----------------------------
    # (1) Node Dropping
    # -----------------------------
    # With probability `drop_node_prob`, remove each node (and its edges).
    # We pick the set of nodes to keep (mask_keep), then call `subgraph`.
    if drop_node_prob > 0.0:
        num_nodes = data_aug.x.size(0)
        # Decide for each node whether to keep it
        mask_keep = torch.rand(num_nodes) >= drop_node_prob
        keep_indices = mask_keep.nonzero(as_tuple=True)[0]
        
        # Subgraph function updates edge_index and preserves only the kept nodes.
        data_aug.edge_index, _ = subgraph(keep_indices, data_aug.edge_index, 
                                          relabel_nodes=True, 
                                          num_nodes=num_nodes)
        
        # Also reduce data_aug.x to only the kept nodes
        if data_aug.x is not None:
            data_aug.x = data_aug.x[keep_indices]

        # If you have other node-level attributes (like data_aug.y, data_aug.pos, etc.),
        # also slice them accordingly:
        # data_aug.y = data_aug.y[keep_indices]  # example if labels per node

    # -----------------------------
    # (2) Edge Dropping
    # -----------------------------
    if drop_edge_prob > 0.0:
        data_aug.edge_index, _ = dropout_edge(
            data_aug.edge_index, 
            p=drop_edge_prob, 
            force_undirected=True, 
            training=True
        )

    # -----------------------------
    # (3) Feature Masking
    # -----------------------------
    if data_aug.x is not None and mask_feature_prob > 0.0:
        # For each entry in data_aug.x, with probability `mask_feature_prob`, set it to zero
        mask = torch.rand_like(data_aug.x) >= mask_feature_prob
        data_aug.x = data_aug.x * mask

    return data_aug

def nt_xent_loss(z1, z2, temperature=0.2):
    """
    Computes an NT-Xent-style contrastive loss for two batches of embeddings.
    z1, z2: [batch_size, dim]
    """
    batch_size = z1.size(0)

    # Normalize embeddings (optional but recommended)
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(z1_norm, z2_norm.T)  # [batch_size, batch_size]
    # Scale by temperature
    sim_matrix = sim_matrix / temperature

    # For each i, the positive pair is (i, i). We'll create the loss
    # so that sim_matrix[i][i] is "the positive" and everything else is "negative".

    # Exponential of similarity
    exp_sim = torch.exp(sim_matrix)

    # For each row i, the denominator is the sum of all exp_sim[i, :]
    denom = torch.sum(exp_sim, dim=1, keepdim=True)  # shape [batch_size, 1]

    # Probability assigned to the positive sample
    pos_sim = torch.diag(exp_sim)  # shape [batch_size]
    loss = -torch.log(pos_sim / (denom.squeeze(1) + 1e-8))

    return loss.mean()