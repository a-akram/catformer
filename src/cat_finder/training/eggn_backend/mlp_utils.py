# Vendored, unmodified, from https://github.com/exatrkx/EggNet (Apache License 2.0)
# Copyright the EggNet / Exa.TrkX authors.
# arXiv:2407.13925 — Calafiura et al., "EggNet: An Evolving Graph-based
# Graph Attention Network for Particle Track Reconstruction"
import torch.nn as nn


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,
    output_layer_norm=False,
    batch_norm=False,
    output_batch_norm=False,
    input_dropout=0,
    hidden_dropout=0,
    track_running_stats=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    for i in range(n_layers - 1):
        if i == 0 and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[i + 1], eps=6e-05, track_running_stats=track_running_stats, affine=True)
            )
        layers.append(hidden_activation())
        if hidden_dropout > 0:
            layers.append(nn.Dropout(hidden_dropout))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if output_layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if output_batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[-1], eps=6e-05, track_running_stats=track_running_stats, affine=True)
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)
