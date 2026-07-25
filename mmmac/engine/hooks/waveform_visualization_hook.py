"""Visualization hook for 1-D time-series classification.

mmpretrain's ``VisualizationHook`` only handles images, so this hook covers
temporal models: every ``interval``-th val/test sample is rendered as a
matplotlib line plot with its ground-truth and predicted class in the title
(green = correct, red = wrong), then stored through the runner's visualizer
backends — ``<work_dir>/<timestamp>/vis_data/vis_image/`` with
``LocalVisBackend``, and wandb when ``WandbVisBackend`` is enabled.

Works with any model whose ``predict`` mode returns
``dict(pred_label=int, gt_label=int)`` per sample (see
:class:`mmmac.models.temporal.LSTMClassifier`).
"""
import math
from typing import Optional, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mmengine.hooks import Hook

from mmmac.registry import HOOKS


@HOOKS.register_module()
class WaveformVisualizationHook(Hook):
    """Render val/test waveforms with their classification results.

    Args:
        enable: Whether the hook draws anything. Defaults to False.
        interval: Visualize every ``interval``-th sample. Defaults to 10.
        classes: Class names for the title. When None, they are taken from
            the dataset's ``metainfo['classes']`` if available, else the
            integer labels are shown.
    """

    def __init__(self,
                 enable: bool = False,
                 interval: int = 10,
                 classes: Optional[Sequence[str]] = None):
        self.enable = enable
        self.interval = interval
        self.classes = classes

    def after_val_iter(self, runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[dict]) -> None:
        self._draw_samples(runner, batch_idx, data_batch, outputs,
                           step=runner.epoch, prefix='val')

    def after_test_iter(self, runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[dict]) -> None:
        self._draw_samples(runner, batch_idx, data_batch, outputs,
                           step=0, prefix='test')

    def _draw_samples(self, runner, batch_idx: int, data_batch: dict,
                      outputs: Sequence[dict], step: int,
                      prefix: str) -> None:
        if not self.enable:
            return

        classes = self.classes or self._classes_from_runner(runner, prefix)
        batch_size = len(outputs)
        start_idx = batch_size * batch_idx
        # first index divisible by the interval, after the start index
        first_sample_id = math.ceil(
            start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, start_idx + batch_size,
                               self.interval):
            idx = sample_id - start_idx
            wave = data_batch['inputs'][idx].detach().cpu().numpy()
            image = self._render(wave, outputs[idx]['gt_label'],
                                 outputs[idx]['pred_label'], classes)
            runner.visualizer.add_image(
                f'{prefix}_waveform_{sample_id}', image, step)

    @staticmethod
    def _classes_from_runner(runner, prefix: str):
        try:
            loop = runner.test_loop if prefix == 'test' else runner.val_loop
            return loop.dataloader.dataset.metainfo.get('classes')
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _render(wave: np.ndarray, gt_label: int, pred_label: int,
                classes: Optional[Sequence[str]]) -> np.ndarray:
        """Plot a (T, C) waveform and return an RGB uint8 array."""

        def name(label: int) -> str:
            if classes is not None and 0 <= label < len(classes):
                return f'{label} ({classes[label]})'
            return str(label)

        correct = gt_label == pred_label
        fig = Figure(figsize=(6, 3), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        for channel in range(wave.shape[-1]):
            ax.plot(wave[:, channel], linewidth=1.2)
        ax.set_title(
            f'GT: {name(gt_label)} / Pred: {name(pred_label)}'
            f' — {"correct" if correct else "WRONG"}',
            color='tab:green' if correct else 'tab:red', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[..., :3].copy()
