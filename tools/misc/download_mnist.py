#!/usr/bin/env python
"""Download MNIST (train + test, ~11 MB) into data/mnist.

The sample configs and the overfit test also download on demand; this script
just lets you prefetch explicitly:

    uv run python tools/misc/download_mnist.py
"""
from mmmac.datasets import MNIST


def main() -> None:
    for split in ('train', 'test'):
        ds = MNIST(data_root='data/mnist', split=split, download=True)
        print(f'{split}: {len(ds)} samples ready under data/mnist')


if __name__ == '__main__':
    main()
