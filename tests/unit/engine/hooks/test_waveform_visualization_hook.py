from types import SimpleNamespace

import numpy as np
import torch
from mmengine.visualization import Visualizer

from mmmac.engine.hooks import WaveformVisualizationHook
from mmmac.registry import HOOKS


def _fake_runner(tmp_path, epoch=3):
    vis = Visualizer(name=f'test-vis-{tmp_path.name}',
                     vis_backends=[dict(type='LocalVisBackend')],
                     save_dir=str(tmp_path))
    return SimpleNamespace(visualizer=vis, epoch=epoch)


def _batch(batch_size=4, seq_len=16):
    data_batch = dict(inputs=torch.randn(batch_size, seq_len, 1))
    outputs = [dict(pred_label=i % 3, gt_label=(i + 1) % 3 if i == 0 else i % 3)
               for i in range(batch_size)]
    return data_batch, outputs


def test_registered_in_mmmac_scope():
    assert HOOKS.get('WaveformVisualizationHook') is WaveformVisualizationHook


def test_disabled_by_default_draws_nothing(tmp_path):
    hook = WaveformVisualizationHook()
    runner = _fake_runner(tmp_path)
    data_batch, outputs = _batch()
    hook.after_val_iter(runner, 0, data_batch, outputs)
    assert not list(tmp_path.rglob('*.png'))


def test_val_and_test_write_pngs_at_interval(tmp_path):
    hook = WaveformVisualizationHook(enable=True, interval=2,
                                     classes=('sine', 'square', 'sawtooth'))
    runner = _fake_runner(tmp_path, epoch=3)
    data_batch, outputs = _batch(batch_size=4)

    hook.after_val_iter(runner, 0, data_batch, outputs)
    hook.after_test_iter(runner, 0, data_batch, outputs)

    vis_dir = tmp_path / 'vis_data' / 'vis_image'
    # samples 0 and 2 of the batch; val is stamped with the epoch, test with 0
    assert (vis_dir / 'val_waveform_0_3.png').exists()
    assert (vis_dir / 'val_waveform_2_3.png').exists()
    assert (vis_dir / 'test_waveform_0_0.png').exists()
    assert (vis_dir / 'test_waveform_2_0.png').exists()


def test_render_returns_rgb_uint8_without_classes():
    wave = np.random.randn(32, 2).astype(np.float32)
    image = WaveformVisualizationHook._render(wave, gt_label=1, pred_label=1,
                                              classes=None)
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[-1] == 3
