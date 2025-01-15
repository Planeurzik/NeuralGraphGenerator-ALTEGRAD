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

# If you still need them:
from new_extract_feats import extract_feats   # numeric extraction for train/valid

from bert_extract_feats import get_combined_feats   # We'll use ONLY this for textual embedding

def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim, device="cpu"):
    """
    Build a list of PyG 'Data' objects for the given dataset split.
    
    We unify the approach so that:
      - train/valid: 
          1) Load adjacency from .graph / BFS reorder
          2) STILL do spectral embeddings or BFS adjacency if you want
          3) Read the textual description, get BERT embedding
          4) Store that as 'stats' in the Data object
      - test:
          1) Only textual description is available, so no adjacency or BFS
          2) Just produce a Data object with 'stats' = BERT embedding.
    
    We are removing references to numeric extraction from 'new_extract_feats.py'.
    """

    data_list = []
    
    # We'll cache the final .pt file so we don't re-run BFS / BERT every time
    save_filename = f"./data/dataset_{dataset}_bert.pt"
    
    if os.path.isfile(save_filename):
        data_list = torch.load(save_filename)
        print(f"Dataset {save_filename} loaded from file")
        return data_list

    # =================== TEST: Only text ===================
    if dataset == 'test':
        desc_file = f"./data/{dataset}/test.txt"
        print(f"[preprocess_dataset] Processing TEST from {desc_file} ...")

        with open(desc_file, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                # Split out graph_id and textual description
                tokens = line.split(",", maxsplit=1)
                graph_id = tokens[0]
                desc_text = tokens[1] if len(tokens) > 1 else ""

                # BERT embedding only, no numeric feats
                bert_feats = get_combined_feats(desc_text, device=device)  # shape [1, embed_dim]
                # Move to CPU if you want Data objects on CPU
                bert_feats = bert_feats.cpu()
                
                # Make a minimal Data object
                data_obj = Data(stats=bert_feats, filename=graph_id)
                data_list.append(data_obj)

        # Save
        torch.save(data_list, save_filename)
        print(f"Dataset {save_filename} saved (test).")

    # =================== TRAIN / VALID: BFS adjacency + BERT ===================
    else:
        graph_path = f"./data/{dataset}/graph"
        desc_path  = f"./data/{dataset}/description"
        print(f"[preprocess_dataset] Processing {dataset.upper()} from {graph_path} + {desc_path} ...")

        files = [f for f in os.listdir(graph_path) 
                   if f.endswith(".graphml") or f.endswith(".edgelist")]

        for f in tqdm(files, desc=f"Building {dataset} data"):
            base, ext = os.path.splitext(f)
            base = os.path.basename(base)  # e.g. "graph_1"
            graph_file = os.path.join(graph_path, f)
            desc_file  = os.path.join(desc_path, base + ".txt")

            # 1) Load the textual description and get BERT feats
            if os.path.isfile(desc_file):
                with open(desc_file, "r", encoding="utf-8") as fd:
                    desc_text = fd.read().strip()
            else:
                desc_text = ""  # no file => empty

            bert_feats = get_combined_feats(desc_text, device=device)
            # shape [1, embed_dim], e.g. [1, 768] if BERT base, or more if you have extra numeric

            # 2) Load the graph, BFS reorder, adjacency
            if ext == ".graphml":
                G = nx.read_graphml(graph_file)
                G = nx.convert_node_labels_to_integers(G, ordering="sorted")
            else:
                # assume .edgelist
                G = nx.read_edgelist(graph_file)
            
            # BFS from largest-degree node in each connected component
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

            node_list_bfs = []
            for comp in CGs:
                # pick largest-degree node
                deg_list = sorted(comp.degree(), key=lambda t: t[1], reverse=True)
                bfs_tree = nx.bfs_tree(comp, source=deg_list[0][0])
                node_list_bfs += list(bfs_tree.nodes())

            adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
            adj_t = torch.from_numpy(adj_bfs).float()

            # spectral embedding if you want
            diags = np.sum(adj_bfs, axis=0)
            D = sparse.diags(diags).toarray()
            L = D - adj_bfs
            with np.errstate(divide="ignore"):
                inv_sqrt = 1.0 / np.sqrt(diags)
            inv_sqrt[np.isinf(inv_sqrt)] = 0
            DH = sparse.diags(diags).toarray()
            L = np.linalg.multi_dot((DH, L, DH))
            L = torch.from_numpy(L).float()

            eigval, eigvecs = torch.linalg.eigh(L)
            eigval = torch.real(eigval)
            eigvecs = torch.real(eigvecs)
            idx_sort = torch.argsort(eigval)
            eigvecs = eigvecs[:, idx_sort]

            edge_index = torch.nonzero(adj_t).t()

            num_nodes = G.number_of_nodes()
            size_diff = n_max_nodes - num_nodes
            # build node features: first col => degree, then spectral
            x = torch.zeros(num_nodes, spectral_emb_dim + 1)
            if (n_max_nodes - 1) > 0:
                # scale degree by (n_max_nodes-1)
                x[:, 0] = torch.mm(adj_t, torch.ones(num_nodes, 1))[:, 0] / (n_max_nodes - 1)

            # take top spectral_emb_dim columns
            max_cols = min(num_nodes, spectral_emb_dim)
            x[:, 1 : (1 + max_cols)] = eigvecs[:, :max_cols]

            # Pad adjacency up to n_max_nodes
            adj_t = F.pad(adj_t, (0, size_diff, 0, size_diff))  
            adj_t = adj_t.unsqueeze(0)  # shape [1, n_max_nodes, n_max_nodes]

            # 3) Build the Data object
            #    - 'stats' holds the BERT embedding 
            data_obj = Data(
                x=x,
                edge_index=edge_index,
                A=adj_t,
                stats=bert_feats,    # we store BERT embedding as stats
                filename=base
            )
            data_list.append(data_obj)

        # Save
        torch.save(data_list, save_filename)
        print(f"Dataset {save_filename} saved ({dataset}).")

    return data_list

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G


def handle_nan(x):
    return float(-100) if math.isnan(x) else x


def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size, L, L, C]
    mask: [batch_size, L, L, 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[1, 2]) / torch.sum(mask, dim=[1, 2])  # [N, C]
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1)) * mask) ** 2
    var = torch.sum(var_term, dim=[1, 2]) / torch.sum(mask, dim=[1, 2])    # [N, C]
    mean = mean.unsqueeze(1).unsqueeze(1)
    var = var.unsqueeze(1).unsqueeze(1)
    instance_norm = (x - mean) / torch.sqrt(var + eps)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size, L, L, C]
    mask: [batch_size, L, L, 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3, 2, 1]) / torch.sum(mask, dim=[3, 2, 1])
    var_term = ((x - mean.view(-1, 1, 1, 1)) * mask) ** 2
    var = torch.sum(var_term, dim=[3, 2, 1]) / torch.sum(mask, dim=[3, 2, 1])
    mean = mean.view(-1, 1, 1, 1)
    var = var.view(-1, 1, 1, 1)
    layer_norm = (x - mean) / torch.sqrt(var + eps)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule from https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
