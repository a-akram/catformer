import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import BatchNorm

from .backbones import BACKBONES


"""
Based on:
Object condensation:
one-stage grid-free multi-object reconstruction in physics detectors, graph and image data
https://arxiv.org/abs/2002.03605

Isabel Haide
Improving ECL Clustering on Trigger Level with Object Condensation

--- Patched for backbone A/B testing ---
Only change from upstream CDCNet: the block loop (GravNetConv-based) has been
moved verbatim into backbones.GravNetBackbone, and a HEPTBackbone alternative
has been added (backbones.HEPTBackbone, wrapping github.com/Graph-COM/HEPT).
Everything from dense_cat onward — including all five output heads and the
preds tensor layout the OC loss reads — is byte-for-byte unchanged.
"""


class CDCNet(nn.Module):
    def __init__(
        self,
        input_dim,
        backbone="gravnet",
        backbone_kwargs=None,
        k=10,
        dim1=64,
        dim2=32,
        nblocks=4,
        coord_dim=2,
        space_dimensions=4,
        momentum=0.6,
    ):
        """
        backbone (str):        "gravnet" (default, matches current behavior) or "hept"
        backbone_kwargs (dict): extra kwargs forwarded to the HEPT backbone
                                 (h_dim, n_layers, num_heads, block_size, n_hashes,
                                 num_regions, num_w_per_dist, dropout). Ignored for gravnet.

        All other args are unchanged from upstream CDCNet.
        """
        super().__init__()

        self.batch_norm_0 = BatchNorm(input_dim, momentum=0.6)

        out_dim = dim2 * nblocks  # unchanged from upstream's dense_cat input width

        if backbone == "gravnet":
            self.backbone = BACKBONES["gravnet"](
                input_dim=input_dim,
                k=k,
                dim1=dim1,
                dim2=dim2,
                nblocks=nblocks,
                space_dimensions=space_dimensions,
                momentum=momentum,
            )
        elif backbone in ("hept", "eggnet"):
            # note: unlike GravNetBackbone, neither HEPT nor EggNet needs the
            # global-exchange pre-step — both do their own global mixing via
            # attention/message-passing — so CDCNet.forward() below is
            # identical for all three backbones.
            self.backbone = BACKBONES[backbone](
                input_dim=input_dim,
                out_dim=out_dim,
                **(backbone_kwargs or {}),
            )
        else:
            raise ValueError(f"Unknown backbone '{backbone}', expected 'gravnet', 'hept', or 'eggnet'")

        # there are skip connections between the blocks (gravnet) / a single
        # readout (hept); this layer is unchanged from upstream either way
        self.dense_cat = nn.Linear(out_dim, dim1)

        # --- output layers: byte-for-byte identical to upstream CDCNet ---
        self.p_beta_layer = nn.Linear(dim1, 1)
        self.p_ccoords_layer = nn.Linear(dim1, coord_dim)
        self.p_p_layer = nn.Linear(dim1, 3)
        self.p_vertex_layer = nn.Linear(dim1, 3)
        self.p_charge_layer = nn.Linear(dim1, 1)

    def forward(self, x, batch):
        x = self.batch_norm_0(x)
        x = self.backbone(x, batch)
        x = F.elu(self.dense_cat(x))

        p_beta = torch.sigmoid(self.p_beta_layer(x))
        p_ccoords = self.p_ccoords_layer(x)
        p_p = self.p_p_layer(x)
        p_vertex = self.p_vertex_layer(x)
        p_charge = torch.sigmoid(self.p_charge_layer(x))

        preds = torch.cat((p_beta, p_ccoords, p_p, p_vertex, p_charge), dim=1)
        return preds
