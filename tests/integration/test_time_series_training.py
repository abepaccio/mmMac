"""End-to-end Runner test on the synthetic time-series config.

Runs a shortened version of configs/time_series/lstm_waveform.py: verifies
config loading, cross-scope registry resolution, the train/val loops and
checkpointing — without any dataset download.
"""
import pytest
from mmengine.config import Config

from mmmac.apis import build_runner


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_lstm_waveform_train_and_val(tmp_path, configs_dir):
    cfg = Config.fromfile(str(configs_dir / 'time_series/lstm_waveform.py'))
    cfg.train_dataloader.dataset.num_samples = 60
    cfg.val_dataloader.dataset.num_samples = 30
    cfg.train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=2)

    runner = build_runner(cfg, work_dir=str(tmp_path))
    runner.train()

    assert (tmp_path / 'epoch_2.pth').exists()

    metrics = runner.val()
    assert 'accuracy/top1' in metrics
    assert 0.0 <= metrics['accuracy/top1'] <= 100.0
