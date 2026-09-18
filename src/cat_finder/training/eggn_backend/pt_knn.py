"""
Portable batched k-NN, pure PyTorch (torch.cdist + topk), no compiled
extension dependency. Brute-force O(n^2) per event, which is fine at CDC
hit-count scales (hundreds of hits/event) and makes this backbone testable
in any environment (including CI) without pyg-lib/torch_cluster.

For larger point clouds, swap this for torch_geometric.nn.knn_graph (uses
torch_cluster, already pinned in packages_install.sh) — see PyGKNN below.
Both expose the same get_graph(x, batch, k, loop=False) -> (start, end)
interface so they're interchangeable in EggNetBackbone.
"""
import torch


class TorchKNN:
    """Brute-force batched k-NN in embedding space, respecting event
    boundaries via `batch` (identical semantics to GravNetConv's `batch`
    argument — no cross-event edges)."""

    def get_graph(self, x, batch, k, loop=False):
        starts, ends = [], []
        k_query = k if loop else k + 1
        for ev in torch.unique(batch):
            idx = (batch == ev).nonzero(as_tuple=True)[0]
            n = idx.shape[0]
            if n == 0:
                continue
            k_eff = min(k_query, n)
            xe = x[idx]
            d = torch.cdist(xe, xe)  # [n, n]
            _, nbr_local = torch.topk(d, k=k_eff, largest=False)  # [n, k_eff]
            self_local = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k_eff)
            if not loop:
                keep = nbr_local != self_local
                nbr_local = nbr_local[keep]
                self_local = self_local[keep]
            else:
                nbr_local = nbr_local.reshape(-1)
                self_local = self_local.reshape(-1)
            starts.append(idx[nbr_local])
            ends.append(idx[self_local])
        return torch.cat(starts), torch.cat(ends)


class PyGKNN:
    """Thin wrapper around torch_geometric.nn.knn_graph — use this instead
    of TorchKNN once event sizes get large enough that O(n^2) brute force
    matters; requires torch_cluster, already in packages_install.sh."""

    def get_graph(self, x, batch, k, loop=False):
        from torch_geometric.nn import knn_graph

        edge_index = knn_graph(x, k=k, batch=batch, loop=loop)
        return edge_index[0], edge_index[1]
