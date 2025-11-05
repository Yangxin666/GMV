"""
GVInf_ran.py
Random-view GVInf
"""

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from common import GCN, generate_views, train, compute_accuracy, measure_inference_time, gvmin
from random import sample

# ----------------------------
# Random selection of views
# ----------------------------
def gvsel_random(views, k=5):
    return [v[0] for v in sample(views, min(k, len(views)))]

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
        selected_views = gvsel_random(views, k=5)

        # Build minimized graph Gm
        Gm, _ = gvmin(selected_views, cora, model, device=device)

        # Inference on Gm
        time_view, logits_view = measure_inference_time(model, Gm)
        time_view_list.append(time_view)

        # Accuracy on Gm
        test_nodes = [i for i, n in enumerate(Gm.nodes.tolist()) if cora.test_mask[n]]
        acc_view = (logits_view[test_nodes].argmax(dim=1) ==
                    cora.y[Gm.nodes[test_nodes]]).float().mean().item() if test_nodes else float('nan')
        acc_view_list.append(acc_view)

        print(f"Run {run+1}: Acc_full={acc_full:.4f}, Acc_view={acc_view:.4f}, Time_full={time_full:.6f}s, Time_view={time_view:.6f}s")

    print("\n===== Average Results =====")
    print(f"Full Graph Accuracy: {np.mean(acc_full_list):.4f} ± {np.std(acc_full_list):.4f}")
    print(f"View-based Accuracy: {np.nanmean(acc_view_list):.4f} ± {np.nanstd(acc_view_list):.4f}")
