"""
GVInf_all.py
GVInf union-based view selection
"""

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from common import GCN, generate_views, train, compute_accuracy, measure_inference_time, gvmin

# ----------------------------
# GVSel (union-based)
# ----------------------------
def gvsel(data, views, test_node, L, k):
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
    mem_view_list = []

    for run in range(runs):
        time_full, logits_full = measure_inference_time(model, cora)
        acc_full = compute_accuracy(logits_full, cora.y, cora.test_mask.to(device))
        acc_full_list.append(acc_full)
        time_full_list.append(time_full)

        views = generate_views(cora, num_views=500, max_hops=3)
        selected_views = gvsel(cora, views, test_node=0, L=2, k=5)

        Gm, partitions = gvmin(selected_views, cora, model, device=device)
        mem_view_list.append(Gm.x.element_size() * Gm.x.nelement() / (1024**2))  # MB

        time_view, logits_view = measure_inference_time(model, Gm)
        test_nodes = [i for i, n in enumerate(Gm.nodes.tolist()) if cora.test_mask[n]]
        acc_view = (logits_view[test_nodes].argmax(dim=1) ==
                    cora.y[Gm.nodes[test_nodes]]).float().mean().item() if test_nodes else float('nan')
        acc_view_list.append(acc_view)
        time_view_list.append(time_view)

        print(f"Run {run+1}: Acc_full={acc_full:.4f}, Acc_view={acc_view:.4f}, Time_full={time_full:.6f}s, Time_view={time_view:.6f}s")
