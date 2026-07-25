"""Top-1 accuracy for models whose ``predict`` mode returns plain dicts.

Works with any model that yields ``dict(pred_label=int, gt_label=int)`` per
sample (e.g. :class:`mmmac.models.temporal.LSTMClassifier`).
"""
from typing import Sequence

from mmengine.evaluator import BaseMetric

from mmmac.registry import METRICS


@METRICS.register_module()
class SimpleAccuracy(BaseMetric):

    default_prefix = 'accuracy'

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            self.results.append(
                int(sample['pred_label'] == sample['gt_label']))

    def compute_metrics(self, results: list) -> dict:
        return dict(top1=100.0 * sum(results) / max(len(results), 1))
