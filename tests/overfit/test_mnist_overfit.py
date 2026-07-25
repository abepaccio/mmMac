"""Overfit sanity check for the MNIST classification recipe.

Trains configs/mnist/lenet5_mnist_overfit.py — LeNet5 memorizing 64 MNIST
images — and asserts near-perfect accuracy on those same samples. This is
the canonical "can this repo actually learn?" test: it exercises the config
system, registries across scopes, the data pipeline, the optimizer and the
val loop with real data.

MNIST (~11 MB) is downloaded to data/mnist on first run.
"""
import pytest
from mmengine.config import Config

from mmmac.apis import build_runner


@pytest.mark.overfit
@pytest.mark.timeout(1800)
def test_lenet5_overfits_64_mnist_samples(tmp_path, repo_root, configs_dir):
    cfg = Config.fromfile(str(configs_dir / 'mnist/lenet5_mnist_overfit.py'))

    # keep the dataset in the repo-level data/ dir so the download is reused
    data_root = str(repo_root / 'data/mnist')
    for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        cfg[loader].dataset.data_root = data_root

    runner = build_runner(cfg, work_dir=str(tmp_path))
    runner.train()

    metrics = runner.val()
    assert metrics['accuracy/top1'] >= 99.0, (
        f'model failed to overfit 64 samples: {metrics}')
