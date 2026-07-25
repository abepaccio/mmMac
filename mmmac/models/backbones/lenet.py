"""Example of extending an mmpretrain model inside the ``mmmac`` scope.

This mirrors the pattern described in the repository docs: import a model
from an upstream OpenMMLab library, subclass it, and register it into the
``mmmac`` registry so configs can refer to it as ``type='mmmac.CustomLeNet5'``.
"""
import torch
import torch.nn.functional as F
from mmpretrain.models.backbones import LeNet5

from mmmac.registry import MODELS


@MODELS.register_module()
class CustomLeNet5(LeNet5):
    """LeNet5 that also accepts raw 28x28 MNIST images.

    The original mmpretrain LeNet5 expects 32x32 inputs; this variant
    zero-pads 28x28 inputs so the MNIST pipeline needs no Resize transform.
    """

    def forward(self, x: torch.Tensor):
        if x.shape[-2:] == (28, 28):
            x = F.pad(x, (2, 2, 2, 2))
        return super().forward(x)
