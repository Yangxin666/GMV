import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import numpy as np
import psutil
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

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
# GVSel with forgetting
# ----------------------------
def gvsel_with_forgetting(data, views, test_node, L, k, decay=0.9, window=None):
    N, _, _, _ = k_hop_subgraph(test_node, L, data.edge_index, relabel_nodes=True)
    N_set = set(N.tolist())

    if window is not None and len(views) > window:
        views = views[-window:]

    S = {}
    num_views = len(views)
    for i, (V, target_nodes, _) in enumerate(views):
        weight = decay ** (num_views - i - 1)
        S[i] = (set(target_nodes), weight)

    selected_views = []
    covered = set()
    while len(covered) < len(N_set) and len(selected_views) < k and S:
        best_view = max(
            S.keys(),
            key=lambda j: len(S[j][0] - covered) * S[j][1]
        )
        selected_views.append(views[best_view][0])
        covered |= S[best_view][0]
        del S[best_view]
    return selected_views

# ----------------------------
# GVMin
# ----------------------------
def gvmin(selected_views, original_data, model):
    partitions = []
    for view in selected_views:
        out = model(view.x, view.edge_index)
        partitions.append(out.mean(dim=0).detach())

    all_global_nodes = torch.cat([v.nodes for v in selected_views], dim=0)
    union_nodes, inverse_indices = torch.unique(all_global_nodes, sorted=True, return_inverse=True)
    global_to_local = {int(g.item()): idx for idx, g in enumerate(union_nodes)}

    Gm_x = original_data.x[union_nodes]

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
        edge_cols = edge_index_tensor.t()
        unique_cols = torch.unique(edge_cols, dim=0)
        edge_index_final = unique_cols.t().contiguous()
    else:
        edge_index_final = torch.empty((2, 0), dtype=torch.long)

    Gm = Data(x=Gm_x, edge_index=edge_index_final)
    Gm.nodes = union_nodes
    return Gm, partitions

# ----------------------------
# Accuracy
# ----------------------------
def compute_accuracy(logits, y, mask):
    preds = logits.argmax(dim=1)
    correct = preds[mask].eq(y[mask]).sum().item()
    return correct / mask.sum().item()

# ----------------------------
# Inference time
# ----------------------------
def measure_inference_time(model, data):
    start = time.time()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    end = time.time()
    return end - start, out

# ----------------------------
# Memory usage
def get_memory_usage(obj=None, device='cpu'):
    """Return memory usage in MB for Data object or overall GPU memory."""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 2)

    if obj is not None:
        def tensor_bytes(o):
            if isinstance(o, torch.Tensor):
                return o.element_size() * o.nelement()
            elif isinstance(o, (list, tuple)):
                return sum(tensor_bytes(x) for x in o)
            elif isinstance(o, dict):
                return sum(tensor_bytes(v) for v in o.values())
            elif hasattr(o, "__dict__"):
                return sum(tensor_bytes(v) for v in o.__dict__.values())
            return 0

        total_bytes = tensor_bytes(obj)
        return total_bytes / (1024 ** 2)

    # fallba


# ----------------------------
# Training
# ----------------------------
def train(model, data, optimizer, epochs=200):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    cora = dataset[0]
    cora.y = torch.squeeze(cora.y)

    # Initialize 2-layer GCN
    model = GCN(in_channels=cora.num_node_features, hidden_channels=32, out_channels=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train model once
    model = train(model, cora, optimizer, epochs=200)
    print("Training complete.")

    runs = 100
    acc_full_list = []
    acc_view_list = []
    time_full_list = []
    time_view_list = []
    mem_view_list = []

    for run in range(runs):
        # Full-graph inference
        time_full, logits_full = measure_inference_time(model, cora)
        acc_full = compute_accuracy(logits_full, cora.y, cora.test_mask)
        acc_full_list.append(acc_full)
        time_full_list.append(time_full)

        # Generate views
        views = generate_views(cora, num_views=500, max_hops=3)
        selected_views = gvsel_with_forgetting(cora, views, test_node=0, L=2, k=5, decay=0.9, window=100)

        # View-based inference
        Gm, partitions = gvmin(selected_views, cora, model)

        # Measure memory usage of minimized view V_m
        mem_view = get_memory_usage(Gm)
        mem_view_list.append(mem_view)
        print(f"Run {run+1}: Memory of single minimized view V_m = {mem_view:.2f} MB")

        time_view, logits_view = measure_inference_time(model, Gm)

        # Map test nodes in minimized graph
        test_nodes = [i for i, n in enumerate(Gm.nodes.tolist()) if cora.test_mask[n]]
        if test_nodes:
            acc_view = (logits_view[test_nodes].argmax(dim=1) ==
                        cora.y[Gm.nodes[test_nodes]]).float().mean().item()
        else:
            acc_view = float('nan')
        acc_view_list.append(acc_view)
        time_view_list.append(time_view)

    # =======================
    # Summary
    # =======================
    print("\n===== Average Results over 100 runs =====")
    print(f"Full Graph Accuracy:       {np.mean(acc_full_list):.4f} ± {np.std(acc_full_list):.4f}")
    print(f"View-based Graph Accuracy: {np.nanmean(acc_view_list):.4f} ± {np.nanstd(acc_view_list):.4f}")
    print(f"Full Graph Inference Time: {np.mean(time_full_list):.6f} sec")
    print(f"View-based Inference Time: {np.mean(time_view_list):.6f} sec")
    print(f"Average Speedup:           {np.mean(np.array(time_full_list)/np.array(time_view_list)):.2f}x")
    print(f"Average Memory of V_m:     {np.mean(mem_view_list):.2f} MB ± {np.std(mem_view_list):.2f}")
