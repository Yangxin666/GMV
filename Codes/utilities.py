# common.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph

# ----------------------------
# 2-layer GCN
# ----------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# ----------------------------
# Generate random views
# ----------------------------
def generate_views(data, num_views=500, max_hops=3):
    views = []
    for _ in range(num_views):
        num_targets = random.randint(1, 5)
        target_nodes = random.sample(range(data.num_nodes), num_targets)
        num_hops = random.randint(1, max_hops)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=target_nodes,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True
        )
        G_V = Data(
            x=data.x[subset],
            edge_index=edge_index,
            y=data.y[subset] if data.y is not None else None
        )
        G_V.nodes = subset.clone()
        views.append((G_V, target_nodes, mapping))
    return views

# ----------------------------
# Training
# ----------------------------
def train(model, data, optimizer, epochs=200, device='cpu'):
    model = model.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

# ----------------------------
# Accuracy
# ----------------------------
def compute_accuracy(logits, y, mask):
    preds = logits.argmax(dim=1)
    correct = preds[mask].eq(y[mask]).sum().item()
    return correct / mask.sum().item()

# ----------------------------
# Inference
# ----------------------------
def measure_inference_time(model, data):
    start = time.time()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    end = time.time()
    return end - start, out

def measure_inference(model, data, device='cpu'):
    model = model.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    if hasattr(data, 'nodes'):
        data.nodes = data.nodes.to(device)
    if hasattr(data, 'y') and data.y is not None:
        data.y = data.y.to(device)

    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    end = time.time()

    if device.startswith('cuda'):
        mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        mem = (data.x.element_size() * data.x.nelement() +
               data.edge_index.element_size() * data.edge_index.nelement()) / (1024 ** 2)
    return end - start, out, mem

# ----------------------------
# GVMin helper
# ----------------------------
def gvmin(selected_views, original_data, model, device='cpu'):
    partitions = []
    for view in selected_views:
        view = view.to(device)
        out = model(view.x, view.edge_index)
        partitions.append(out.mean(dim=0).detach())

    all_global_nodes = torch.cat([v.nodes for v in selected_views], dim=0)
    union_nodes, inverse_indices = torch.unique(all_global_nodes, sorted=True, return_inverse=True)
    global_to_local = {int(g.item()): idx for idx, g in enumerate(union_nodes)}

    Gm_x = original_data.x[union_nodes].to(device)

    edge_pairs = []
    for view in selected_views:
        if view.edge_index.numel() == 0:
            continue
        src_global = [int(view.nodes[i].item()) for i in view.edge_index[0].tolist()]
        dst_global = [int(view.nodes[i].item()) for i in view.edge_index[1].tolist()]
        for g_s, g_d in zip(src_global, dst_global):
            edge_pairs.append((global_to_local[g_s], global_to_local[g_d]))

    if edge_pairs:
        edge_index_tensor = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        unique_cols = torch.unique(edge_index_tensor.t(), dim=0)
        edge_index_final = unique_cols.t().contiguous().to(device)
    else:
        edge_index_final = torch.empty((2, 0), dtype=torch.long, device=device)

    Gm = Data(x=Gm_x, edge_index=edge_index_final)
    Gm.nodes = union_nodes.to(device)
    return Gm, partitions
