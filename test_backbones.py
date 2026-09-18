"""
Smoke test for the GravNet/HEPT backbone swap in CDCNet.

Run from the repo root (after applying the patch) with:
    pip install einops --break-system-packages   # if not already present
    python test_backbones.py

This uses synthetic data shaped like real CDC hits (7 input features, the
first two being cdchit_middle_x / cdchit_middle_y) spread across a few
"events" in one minibatch, exactly like a batch coming out of CDCDataset +
PyG's DataLoader. It does NOT require basf2 or any real dataset — it only
checks that both backbones (a) run forward without shape errors, (b) produce
a preds tensor of the shape the OC loss expects, and (c) are differentiable.
"""
import torch
from cat_finder.training.gnn_model import CDCNet

torch.manual_seed(0)


def make_fake_batch(n_events=3, hits_per_event=(40, 55, 30), input_dim=7):
    xs, batch_idx = [], []
    for ev, n_hits in enumerate(hits_per_event[:n_events]):
        feats = torch.randn(n_hits, input_dim)
        feats[:, 0] = torch.randn(n_hits) * 10.0  # cdchit_middle_x
        feats[:, 1] = torch.randn(n_hits) * 10.0  # cdchit_middle_y
        xs.append(feats)
        batch_idx.append(torch.full((n_hits,), ev, dtype=torch.long))
    return torch.cat(xs, dim=0), torch.cat(batch_idx, dim=0)


def run_case(backbone, backbone_kwargs=None):
    print(f"\n=== backbone = {backbone} ===")
    x, batch = make_fake_batch()
    n_hits, input_dim = x.shape

    net = CDCNet(
        input_dim=input_dim,
        backbone=backbone,
        backbone_kwargs=backbone_kwargs or {},
        k=8,
        dim1=32,
        dim2=16,
        nblocks=2,
        coord_dim=2,
        space_dimensions=4,
        momentum=0.6,
    ).float()

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    preds = net(x, batch)
    expected_width = 1 + 2 + 3 + 3 + 1  # beta, ccoords(2), p(3), vertex(3), charge
    assert preds.shape == (n_hits, expected_width), preds.shape
    print(f"preds shape OK: {tuple(preds.shape)}")

    # differentiability check
    loss = preds.pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm() for p in net.parameters() if p.grad is not None)
    assert grad_norm > 0
    print(f"backward() OK, total grad norm: {grad_norm:.4f}")


if __name__ == "__main__":
    run_case("gravnet")
    run_case(
        "hept",
        backbone_kwargs=dict(
            h_dim=32,
            n_layers=2,
            num_heads=2,
            block_size=16,   # small on purpose: synthetic events are tiny
            n_hashes=3,
            num_regions=8,
            num_w_per_dist=6,
            dropout=0.0,
        ),
    )
    run_case(
        "eggnet",
        backbone_kwargs=dict(
            node_rep_dim=32,
            edge_rep_dim=16,
            n_iters=2,
            n_gnns_per_iter=2,
            knn_train=8,
        ),
    )
    print("\nAll smoke tests passed.")
