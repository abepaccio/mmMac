import torch
from mmpretrain.models.backbones import LeNet5

from mmmac.models.backbones import CustomLeNet5
from mmmac.registry import MODELS


def test_registered_in_mmmac_scope():
    assert MODELS.get('CustomLeNet5') is CustomLeNet5


def test_is_mmpretrain_subclass():
    assert issubclass(CustomLeNet5, LeNet5)


def test_forward_accepts_raw_mnist_28x28():
    model = CustomLeNet5(num_classes=10)
    out = model(torch.randn(2, 1, 28, 28))
    assert out[-1].shape == (2, 10)


def test_forward_still_accepts_32x32():
    model = CustomLeNet5(num_classes=10)
    out = model(torch.randn(2, 1, 32, 32))
    assert out[-1].shape == (2, 10)
