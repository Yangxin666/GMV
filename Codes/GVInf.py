# gvinf_new_accuracy.py
from common import *
import torch
from torch_geometric.datasets import Planetoid

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

if __name__ == "__main__":
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    cora = dataset[0]
    cora.y = torch.squeeze(cora.y)

    model = GCN(cora.num_node_features, 32, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model = train(model, cora, optimizer, epochs=200)

    # Continue with run loop, view generation, gvsel_with_forgetting, gvmin, and evaluation
