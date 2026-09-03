"""Compatibility shim for the model base class.

The released Python demo only needs ``torch.nn.Module`` functionality at
inference time. Importing PyTorch Lightning for that path pulls in torchmetrics
and a large dependency graph before the first model is loaded. Training keeps
the original Lightning base class; the demo opts into the lightweight base by
setting ``MOTIONBRICKS_LIGHTWEIGHT_INFERENCE=1`` before model imports.
"""

from __future__ import annotations

import os

import torch


def get_lightning_module():
    """Return the training or inference base class without changing APIs."""

    if os.environ.get("MOTIONBRICKS_LIGHTWEIGHT_INFERENCE") == "1":
        return torch.nn.Module

    # Keep this import out of the inference process. In training this branch
    # retains LightningModule semantics and the existing trainer integration.
    from pytorch_lightning import LightningModule

    return LightningModule


LightningModule = get_lightning_module()

