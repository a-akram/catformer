import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GravNetConv, BatchNorm, global_mean_pool

from .hept_backend.transformer import Transformer
from .eggn_backend.eggnet_core import EggNetCore


class GravNetBackbone(nn.Module):
    """Exact extraction of CDCNet's existing block loop (today: self.blocks +
    the global-exchange step in forward()). Behavior is unchanged from the
    current CDCNet — only moved into its own module so it can be swapped.

    forward(x, batch) -> Tensor[N_hits, dim2 * nblocks]
    """

    def __init__(self, input_dim, k=10, dim1=64, dim2=32, nblocks=4, space_dimensions=4, momentum=0.6):
        super().__init__()
        self.out_dim = dim2 * nblocks

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(2 * input_dim, dim1),
                        nn.Linear(dim1, dim1),
                        BatchNorm(dim1, momentum=momentum),
                        nn.Linear(dim1, dim1),
                        GravNetConv(
                            in_channels=dim1,
                            out_channels=dim1 * 2,
                            space_dimensions=space_dimensions,
                            k=k,
                            propagate_dimensions=dim1,
                        ),
                        BatchNorm(dim1 * 2, momentum=momentum),
                        nn.Linear(dim1 * 2, dim2),
                    ]
                )
            ]
        )
        self.blocks.extend(
            nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Linear(4 * dim1, dim1),
                            nn.Linear(dim1, dim1),
                            BatchNorm(dim1, momentum=momentum),
                            nn.Linear(dim1, dim1),
                            GravNetConv(
                                in_channels=dim1,
                                out_channels=dim1 * 2,
                                space_dimensions=space_dimensions,
                                k=k,
                                propagate_dimensions=dim1,
                            ),
                            BatchNorm(dim1 * 2, momentum=momentum),
                            nn.Linear(dim1 * 2, dim2),
                        ]
                    )
                    for _ in range(nblocks - 1)
                ]
            )
        )

    def forward(self, x, batch):
        feat = []

        # global exchange (identical to current CDCNet.forward)
        out = global_mean_pool(x, batch)
        x = torch.cat([x, out[batch]], dim=-1)

        for i, block in enumerate(self.blocks):
            if i > 0:
                out = global_mean_pool(x, batch)
                x = torch.cat([x, out[batch]], dim=-1)
            x = F.elu(block[0](x))
            x = F.elu(block[1](x))
            x = block[2](x)
            x = F.elu(block[3](x))
            x = block[4](x, batch)
            x = block[5](x)
            feat.append(F.elu(block[6](x)))

        return torch.cat(feat, dim=1)  # [N_hits, dim2 * nblocks]


class HEPTBackbone(nn.Module):
    """Thin adapter around the vendored HEPT Transformer (Miao et al., ICML 2024).
    Reuses the same coordinate columns CATFinder already puts first in its
    feature tensor (cdchit_middle_x, cdchit_middle_y) as the LSH region
    coordinates. Multi-event batching is handled natively by the vendored
    code via the `batch` tensor (see hept_backend/transformer.py::prepare_input),
    no extra plumbing needed here.

    forward(x, batch) -> Tensor[N_hits, out_dim]
    """

    def __init__(
        self,
        input_dim,
        out_dim,
        coords_dim=2,
        h_dim=64,
        n_layers=4,
        num_heads=4,
        block_size=100,
        n_hashes=3,
        num_regions=100,
        num_w_per_dist=10,
        dropout=0.1,
    ):
        super().__init__()
        self.coords_dim = coords_dim
        self.net = Transformer(
            in_dim=input_dim,
            coords_dim=coords_dim,
            num_classes=out_dim,
            h_dim=h_dim,
            n_layers=n_layers,
            num_heads=num_heads,
            block_size=block_size,
            n_hashes=n_hashes,
            num_regions=num_regions,
            num_w_per_dist=num_w_per_dist,
            dropout=dropout,
        )

    def forward(self, x, batch):
        coords = x[:, : self.coords_dim]  # cdchit_middle_x, cdchit_middle_y
        return self.net(x, coords, batch)


class EggNetBackbone(nn.Module):
    """Thin adapter around the adapted EggNet core (Calafiura et al.,
    arXiv:2407.13925 / github.com/exatrkx/EggNet, Apache-2.0). See
    eggnet_backend/eggnet_core.py's docstring for exactly what was changed
    from upstream and why.

    forward(x, batch) -> Tensor[N_hits, out_dim]
    """

    def __init__(self, input_dim, out_dim, **eggnet_kwargs):
        super().__init__()
        self.net = EggNetCore(input_dim=input_dim, out_dim=out_dim, **eggnet_kwargs)

    def forward(self, x, batch):
        return self.net(x, batch)


BACKBONES = {
    "gravnet": GravNetBackbone,
    "hept": HEPTBackbone,
    "eggnet": EggNetBackbone,
}
