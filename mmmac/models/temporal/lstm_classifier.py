"""Time-series model example built directly on mmengine's BaseModel.

Time-series tasks have no dedicated OpenMMLab library, so this module shows
the intended extension path: implement ``forward(inputs, labels, mode)`` on
``mmengine.model.BaseModel`` and the mmengine Runner (loops, hooks, wandb
logging, checkpointing) works unchanged.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel

from mmmac.registry import MODELS


@MODELS.register_module()
class LSTMClassifier(BaseModel):
    """Classify a (batch, time, channels) sequence with an LSTM encoder."""

    def __init__(self,
                 in_channels: int = 1,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 num_classes: int = 3,
                 pool: str = 'mean',
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        assert pool in ('mean', 'last'), pool
        # 'mean' pools the LSTM outputs over time, which trains far more
        # robustly across random inits than reading only the last step
        self.pool = pool
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers,
                            batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self,
                inputs: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mode: str = 'tensor'):
        feats, _ = self.lstm(inputs)
        feats = feats.mean(dim=1) if self.pool == 'mean' else feats[:, -1]
        logits = self.head(feats)

        if mode == 'loss':
            return dict(loss=F.cross_entropy(logits, labels))
        if mode == 'predict':
            preds = logits.argmax(dim=1)
            if labels is None:
                labels = torch.full_like(preds, -1)
            return [
                dict(pred_label=int(p), gt_label=int(g))
                for p, g in zip(preds, labels)
            ]
        return logits
