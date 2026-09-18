"""
Adapted from https://github.com/exatrkx/EggNet (Apache License 2.0),
eggnet/models/eggnet.py — Calafiura et al., arXiv:2407.13925.

Changes from upstream EggNet, as required to be noted under Apache-2.0:
  1. forward(x, batch) takes a plain feature tensor + event-index tensor,
     matching CDCNet's backbone contract — upstream reads named features
     off a mutated PyG-Batch-like object (`batch[feature_name]`).
  2. Returns the final embedding tensor directly, instead of writing it
     back onto `batch.hit_embedding`.
  3. The ~25-key upstream `hparams` dict is reduced to the subset needed
     for this comparison; `node_filter`, `recurrent_gnn`, and gradient
     checkpointing are dropped (hardcoded off) to shrink the config surface.
  4. k-NN backend is pluggable (pt_knn.TorchKNN by default, or PyGKNN);
     upstream defaults to RAPIDS cuml (`cu_knn`) or faiss, neither of which
     CATFinder otherwise depends on.
  5. `recurrent=True` behavior only (matches EggNet's own recommended/
     default configs) — the non-recurrent per-iteration-weights branch is
     not carried over, again to shrink the config surface for this
     comparison; reinstate it by copying more of upstream's ModuleList
     indexing logic if you need it.

The message-passing math itself (gat / message_passing) is unchanged from
upstream — that's the actual EggNet architecture being benchmarked.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import softmax

try:
    from torch_scatter import scatter_add
except ImportError:
    # Pure-PyTorch fallback, equivalent to torch_scatter.scatter_add for the
    # dim=0 case used here. CATFinder's packages_install.sh already installs
    # the real torch_scatter, so this only matters for environments (e.g.
    # quick CPU-only smoke tests) where the compiled extension isn't set up.
    def scatter_add(src, index, dim=0, dim_size=None):
        assert dim == 0
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        out = src.new_zeros((dim_size,) + src.shape[1:])
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        return out.scatter_add_(0, idx, src)

from .mlp_utils import make_mlp
from .pt_knn import TorchKNN


class EggNetCore(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        node_rep_dim=64,
        edge_rep_dim=64,
        n_iters=2,
        n_gnns_per_iter=2,
        encoder_hidden=64,
        node_0_hidden=64,
        edge_hidden=64,
        node_hidden=64,
        n_encoder_layers=2,
        n_node_0_layers=2,
        n_edge_layers=2,
        n_node_layers=2,
        hidden_activation="ReLU",
        knn_train=10,
        embedding_norm=False,
        knn_backend=None,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.n_gnns_per_iter = n_gnns_per_iter
        self.knn_train = knn_train
        self.node_rep_dim = node_rep_dim
        self.embedding_norm = embedding_norm
        self.knn = knn_backend or TorchKNN()

        self.node_encoder = make_mlp(
            input_dim,
            [encoder_hidden] * (n_encoder_layers - 1) + [node_rep_dim],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
        )
        self.node_network_0 = make_mlp(
            node_rep_dim,
            [node_0_hidden] * (n_node_0_layers - 1) + [node_rep_dim],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
        )

        n_edge_nets = n_iters * n_gnns_per_iter
        self.edge_networks = nn.ModuleList(
            [
                make_mlp(
                    node_rep_dim * 2 if j == 0 else node_rep_dim * 2 + edge_rep_dim,
                    [edge_hidden] * (n_edge_layers - 1) + [edge_rep_dim + 1],
                    hidden_activation=hidden_activation,
                    output_activation=hidden_activation,
                )
                for _ in range(n_iters)
                for j in range(n_gnns_per_iter)
            ]
        )
        self.node_networks = nn.ModuleList(
            [
                make_mlp(
                    node_rep_dim + edge_rep_dim,
                    [node_hidden] * (n_node_layers - 1) + [node_rep_dim],
                    hidden_activation=hidden_activation,
                    output_activation=hidden_activation,
                )
                for _ in range(n_iters * n_gnns_per_iter)
            ]
        )

        # readout to a fixed width, matching whatever dense_cat expects
        self.readout = nn.Linear(node_rep_dim, out_dim)

    def build_edges(self, x, batch):
        emb = x.detach()
        if self.embedding_norm:
            emb = F.normalize(emb)
        return self.knn.get_graph(emb, batch, k=self.knn_train, loop=False)

    def gat(self, x, start, end, i):
        e = None
        for j in range(self.n_gnns_per_iter):
            x, e = self.message_passing(e, x, start, end, i, j)
        return x

    def message_passing(self, e, x, start, end, i, j):
        # unchanged from upstream EggNet.message_passing
        e = torch.cat([x[start], x[end]] if j == 0 else [x[start], x[end], e], dim=-1)
        net_idx = i * self.n_gnns_per_iter + j
        e = self.edge_networks[net_idx](e)
        w = e[:, -1:]
        w = softmax(w, end)
        e = e[:, :-1]

        w = scatter_add(e * w, end, dim=0, dim_size=x.shape[0])
        x = torch.cat([x, w], dim=1)
        x = self.node_networks[net_idx](x)
        return x, e

    def forward(self, x, batch):
        v = self.node_encoder(x)
        x = self.node_network_0(v)

        for i in range(self.n_iters):
            start, end = self.build_edges(x, batch)
            x = self.gat(x, start, end, i)

        out = self.readout(x)
        if self.embedding_norm:
            out = F.normalize(out)
        return out
