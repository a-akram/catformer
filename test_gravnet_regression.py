"""
Regression test: proves GravNetBackbone (the refactored code) produces
BIT-IDENTICAL output to the original, pre-refactor CDCNet for the same
weights and inputs. This is the evidence a reviewer needs that moving the
block loop into its own module changed nothing about GravNet's behavior —
only HEPT/EggNet are new; GravNet's math is untouched.

NOTE: requires pyg-lib (for GravNetConv), same as the rest of the repo
already does. Not runnable in a sandbox without it — run this in your real
CATFinder environment before opening the merge request.

Usage:
    PYTHONPATH=src python test_gravnet_regression.py
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GravNetConv, BatchNorm, global_mean_pool

from cat_finder.training.gnn_model import CDCNet


class OriginalCDCNet(nn.Module):
    """Verbatim copy of upstream CDCNet, before the backbone refactor.
    Used only as a reference to diff against — not part of the new code."""

    def __init__(self, input_dim, k=10, dim1=64, dim2=32, nblocks=4, coord_dim=2,
                 space_dimensions=4, momentum=0.6):
        super().__init__()
        self.batch_norm_0 = BatchNorm(input_dim, momentum=0.6)
        self.blocks = nn.ModuleList([nn.ModuleList([
            nn.Linear(2 * input_dim, dim1), nn.Linear(dim1, dim1),
            BatchNorm(dim1, momentum=momentum), nn.Linear(dim1, dim1),
            GravNetConv(in_channels=dim1, out_channels=dim1 * 2, space_dimensions=space_dimensions,
                        k=k, propagate_dimensions=dim1),
            BatchNorm(dim1 * 2, momentum=momentum), nn.Linear(dim1 * 2, dim2),
        ])])
        self.blocks.extend(nn.ModuleList([nn.ModuleList([
            nn.Linear(4 * dim1, dim1), nn.Linear(dim1, dim1),
            BatchNorm(dim1, momentum=momentum), nn.Linear(dim1, dim1),
            GravNetConv(in_channels=dim1, out_channels=dim1 * 2, space_dimensions=space_dimensions,
                        k=k, propagate_dimensions=dim1),
            BatchNorm(dim1 * 2, momentum=momentum), nn.Linear(dim1 * 2, dim2),
        ]) for _ in range(nblocks - 1)]))
        self.dense_cat = nn.Linear(dim2 * nblocks, dim1)
        self.p_beta_layer = nn.Linear(dim1, 1)
        self.p_ccoords_layer = nn.Linear(dim1, coord_dim)
        self.p_p_layer = nn.Linear(dim1, 3)
        self.p_vertex_layer = nn.Linear(dim1, 3)
        self.p_charge_layer = nn.Linear(dim1, 1)

    def forward(self, x, batch):
        feat = []
        x = self.batch_norm_0(x)
        out = global_mean_pool(x, batch)
        x = torch.cat([x, out[batch]], dim=-1)
        for i, block in enumerate(self.blocks):
            if i > 0:
                out = global_mean_pool(x, batch)
                x = torch.cat([x, out[batch]], dim=-1)
            x = F.elu(block[0](x)); x = F.elu(block[1](x)); x = block[2](x)
            x = F.elu(block[3](x)); x = block[4](x, batch); x = block[5](x)
            feat.append(F.elu(block[6](x)))
        x = torch.cat(feat, dim=1)
        x = F.elu(self.dense_cat(x))
        p_beta = torch.sigmoid(self.p_beta_layer(x))
        p_ccoords = self.p_ccoords_layer(x)
        p_p = self.p_p_layer(x)
        p_vertex = self.p_vertex_layer(x)
        p_charge = torch.sigmoid(self.p_charge_layer(x))
        return torch.cat((p_beta, p_ccoords, p_p, p_vertex, p_charge), dim=1)


def copy_weights_original_to_refactored(orig: OriginalCDCNet, new: CDCNet):
    """Map OriginalCDCNet's flat state_dict keys onto the refactored CDCNet's
    backbone.* keys, then load the rest (dense_cat, output heads) directly."""
    sd_orig = orig.state_dict()
    sd_new = new.state_dict()
    remapped = {}
    for k, v in sd_orig.items():
        if k.startswith("blocks."):
            remapped["backbone." + k] = v
        else:
            remapped[k] = v
    missing = set(sd_new.keys()) - set(remapped.keys())
    assert not missing, f"unexpected unmapped keys: {missing}"
    new.load_state_dict(remapped)


def make_fake_batch(n_events=3, hits_per_event=(40, 55, 30), input_dim=7):
    xs, batch_idx = [], []
    for ev, n_hits in enumerate(hits_per_event[:n_events]):
        feats = torch.randn(n_hits, input_dim)
        xs.append(feats)
        batch_idx.append(torch.full((n_hits,), ev, dtype=torch.long))
    return torch.cat(xs, dim=0), torch.cat(batch_idx, dim=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    kwargs = dict(input_dim=7, k=8, dim1=32, dim2=16, nblocks=2, coord_dim=2,
                  space_dimensions=4, momentum=0.6)

    orig = OriginalCDCNet(**kwargs).float().eval()
    new = CDCNet(backbone="gravnet", **kwargs).float().eval()
    copy_weights_original_to_refactored(orig, new)

    x, batch = make_fake_batch(input_dim=kwargs["input_dim"])
    with torch.no_grad():
        out_orig = orig(x, batch)
        out_new = new(x, batch)

    assert torch.equal(out_orig, out_new), "Refactor changed GravNet's output!"
    print("PASS: refactored GravNetBackbone output is bit-identical to upstream CDCNet.")
