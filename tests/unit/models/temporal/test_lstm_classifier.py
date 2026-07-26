import torch

from mmmac.models.temporal import LSTMClassifier
from mmmac.registry import MODELS


def _model() -> LSTMClassifier:
    return LSTMClassifier(in_channels=1, hidden_size=8, num_classes=3)


def test_registered_in_mmmac_scope():
    assert MODELS.get('LSTMClassifier') is LSTMClassifier


def test_tensor_mode_returns_logits():
    logits = _model()(torch.randn(4, 16, 1), mode='tensor')
    assert logits.shape == (4, 3)


def test_last_step_pooling_variant():
    model = LSTMClassifier(in_channels=1, hidden_size=8, num_classes=3,
                           pool='last')
    logits = model(torch.randn(4, 16, 1), mode='tensor')
    assert logits.shape == (4, 3)


def test_loss_mode_returns_scalar_loss():
    labels = torch.tensor([0, 1, 2, 0])
    losses = _model()(torch.randn(4, 16, 1), labels, mode='loss')
    assert losses['loss'].ndim == 0
    assert torch.isfinite(losses['loss'])


def test_predict_mode_returns_per_sample_dicts():
    labels = torch.tensor([0, 1, 2, 0])
    preds = _model()(torch.randn(4, 16, 1), labels, mode='predict')
    assert len(preds) == 4
    assert {'pred_label', 'gt_label'} <= preds[0].keys()
    assert preds[1]['gt_label'] == 1
