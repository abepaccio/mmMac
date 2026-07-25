import torch

from mmmac.datasets import SyntheticWaveformDataset
from mmmac.datasets.time_series import WAVEFORM_CLASSES
from mmmac.registry import DATASETS


def test_registered_in_mmmac_scope():
    assert DATASETS.get('SyntheticWaveformDataset') is SyntheticWaveformDataset


def test_item_shape_and_types():
    ds = SyntheticWaveformDataset(num_samples=12, seq_len=32)
    assert len(ds) == 12
    item = ds[0]
    assert item['inputs'].shape == (32, 1)
    assert item['inputs'].dtype == torch.float32
    assert isinstance(item['labels'], int)


def test_labels_cover_all_classes():
    ds = SyntheticWaveformDataset(num_samples=9)
    labels = {ds[i]['labels'] for i in range(len(ds))}
    assert labels == set(range(len(WAVEFORM_CLASSES)))


def test_deterministic_for_same_seed():
    a = SyntheticWaveformDataset(num_samples=6, seed=42)
    b = SyntheticWaveformDataset(num_samples=6, seed=42)
    assert torch.equal(a[3]['inputs'], b[3]['inputs'])


def test_test_mode_changes_samples():
    train = SyntheticWaveformDataset(num_samples=6, seed=42)
    val = SyntheticWaveformDataset(num_samples=6, seed=42, test_mode=True)
    assert not torch.equal(train[0]['inputs'], val[0]['inputs'])
