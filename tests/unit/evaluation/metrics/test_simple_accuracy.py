from mmmac.evaluation import SimpleAccuracy
from mmmac.registry import METRICS


def test_registered_in_mmmac_scope():
    assert METRICS.get('SimpleAccuracy') is SimpleAccuracy


def test_computes_top1_percentage():
    metric = SimpleAccuracy()
    samples = [
        dict(pred_label=0, gt_label=0),
        dict(pred_label=1, gt_label=1),
        dict(pred_label=2, gt_label=1),
        dict(pred_label=0, gt_label=0),
    ]
    metric.process(data_batch=None, data_samples=samples)
    results = metric.evaluate(size=len(samples))
    assert results['accuracy/top1'] == 75.0
