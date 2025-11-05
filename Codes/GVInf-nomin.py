"""
GVInf_nomin.py
Per-view inference without graph minimization
"""

import torch
import numpy as np
import time
from torch_geometric.datasets import Planetoid
from common import GCN, generate_views, train, compute_accuracy

# ----------------------------
# GVSel greedy
# ----------------------------
def gvsel(data, views, test_node, L, k=5):
    from torch_geometric.utils import k_hop_subgraph
    N, _, _, _ = k_hop_subgraph(test_node, L, data.edge_index, relabel_nodes=True)
    N_set = set(N.tolist())
    S = {i: set(target_nodes) for i, (V, target_nodes, _) in enumerate(views)}

    selected_views = []
    covered = set()
    while len(covered) < len(N_set) and len(selected_views) < k and S:
        best_view = max(S.keys(), key=lambda j: len(S[j] - covered))
        selected_views.append(views[best_view][0])
        covered |= S[best_view]
        del S[best_view]
    return selected_views

# ----------------------------
# Per-view inference & aggregation
# ----------------------------
def inference_on_views(model, selected_views, test_mask):
    import torch
    from collections import defaultdict
    node_embeddings = defaultdict(list)
    total_time = 0.0

    for view in selected_views:
        start = time.time()
        out = model(view.x, view.edge_index)
        total_time += time.time() - start
        for gid, emb in zip(view.nodes.tolist(), out):
            node_embeddings[gid].append(emb.cpu())

    num_classes = out.shape[1]
    agg_logits = torch.zeros((len(test_mask), num_classes))
    count = torch.zeros(len(test_mask))
    for gid, embs in node_embeddings.items():
        stacked = torch.stack(embs, dim=0)
        agg_logits[gid] = stacked.mean(dim=0)
        count[gid] = 1

    mask_indices = torch.where(test_mask)[0]
    final_mask = mask_indices[count[mask_indices] > 0]
    return agg_logits, final_mask, total_time

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    cora = dataset[0]
    cora.y = torch.squeeze(cora.y).to(device)

    model = GCN(in_channels=cora.num_node_features, hidden_channels=32, out_channels=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model = train(model, cora, optimizer, epochs=200, device=device)
    print("Training complete.")

    runs = 100
    acc_full_list, acc_view_list = [], []
    time_full_list, time_view_list = [], []

    for run in range(runs):
        # Full-graph inference
        time_full, logits_full = measure_inference_time(model, cora)
        acc_full = compute_accuracy(logits_full, cora.y, cora.test_mask.to(device))
        acc_full_list.append(acc_full)
        time_full_list.append(time_full)

        # Generate views
        views = generate_views(cora, num_views=500, max_hops=3)
        selected_views = gvsel(cora, views, test_node=0, L=2, k=5)

        # Per-view inference
        agg_logits, final_mask, time_view = inference_on_views(model, selected_views, cora.test_mask.to(device))
        time_view_list.append(time_view)

        # Accuracy
        if len(final_mask) > 0:
            acc_view = (agg_logits[final_mask].argmax(dim=1) == cora.y[final_mask].cpu()).float().mean().item()
        else:
            acc_view = float('nan')
        acc_view_list.append(acc_view)

        print(f"Run {run+1}: Acc_full={acc_full:.4f}, Acc_view={acc_view:.4f}, Time_full={time_full:.6f}s, Time_view={time_view:.6f}s")

    print("\n===== Average Results =====")
    print(f"Full Graph Accuracy: {np.mean(acc_full_list):.4f} ± {np.std(acc_full_list):.4f}")
    print(f"View-based Accuracy: {np.nanmean(acc_view_list):.4f} ± {np.nanstd(acc_view_list):.4f}")
