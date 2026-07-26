"""Deterministic synthetic waveform dataset for time-series classification.

Used by ``configs/time_series/`` and the integration tests. Being fully
synthetic it needs no download and is reproducible from ``seed``.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from mmmac.registry import DATASETS

WAVEFORM_CLASSES = ('sine', 'square', 'sawtooth')


@DATASETS.register_module()
class SyntheticWaveformDataset(Dataset):
    """Classify noisy sine / square / sawtooth waves.

    Each item is ``dict(inputs=FloatTensor(seq_len, 1), labels=int)``, ready
    for ``default_collate`` and :class:`mmmac.models.temporal.LSTMClassifier`.
    """

    METAINFO = dict(classes=WAVEFORM_CLASSES)

    def __init__(self,
                 num_samples: int = 300,
                 seq_len: int = 64,
                 noise: float = 0.1,
                 seed: int = 0,
                 test_mode: bool = False):
        rng = np.random.default_rng(seed + int(test_mode))
        t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)

        data = np.empty((num_samples, seq_len, 1), dtype=np.float32)
        labels = np.empty(num_samples, dtype=np.int64)
        for i in range(num_samples):
            label = i % len(WAVEFORM_CLASSES)
            freq = rng.uniform(1.0, 3.0)
            phase = rng.uniform(0.0, 1.0)
            arg = freq * t + phase
            if label == 0:  # sine
                wave = np.sin(2 * np.pi * arg)
            elif label == 1:  # square
                wave = np.sign(np.sin(2 * np.pi * arg))
            else:  # sawtooth
                wave = 2.0 * (arg % 1.0) - 1.0
            wave = wave + noise * rng.standard_normal(seq_len)
            data[i, :, 0] = wave.astype(np.float32)
            labels[i] = label

        self._data = torch.from_numpy(data)
        self._labels = labels

    @property
    def metainfo(self) -> dict:
        return dict(self.METAINFO)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        return dict(inputs=self._data[idx], labels=int(self._labels[idx]))
