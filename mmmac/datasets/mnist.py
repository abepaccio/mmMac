"""MNIST dataset with a working download mirror and a working ``split``.

Extends :class:`mmpretrain.datasets.MNIST` — same pattern as the model
example in ``mmmac/models/backbones/lenet.py`` — with two fixes:

1. The upstream class downloads from ``yann.lecun.com`` which is frequently
   offline; this subclass points at the torchvision S3 mirror instead.
   File names, md5 checksums and the loading logic are inherited unchanged.
2. Upstream selects train/test files from ``test_mode`` and silently
   ignores ``split``, so ``split='test'`` alone still loads the *training*
   images. Here ``split`` is authoritative: when ``test_mode`` is not
   passed explicitly it is derived from ``split``.
"""
from typing import Optional

from mmpretrain.datasets import MNIST as MMPretrainMNIST

from mmmac.registry import DATASETS


@DATASETS.register_module()
class MNIST(MMPretrainMNIST):

    url_prefix = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

    def __init__(self,
                 data_root: str = '',
                 split: str = 'train',
                 test_mode: Optional[bool] = None,
                 **kwargs):
        if test_mode is None:
            test_mode = split == 'test'
        super().__init__(data_root=data_root, split=split,
                         test_mode=test_mode, **kwargs)
