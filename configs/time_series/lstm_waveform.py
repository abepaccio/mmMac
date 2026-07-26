# Time-series classification example: LSTM on synthetic waveforms.
# Demonstrates that non-image (temporal) models plug into the same Runner —
# see docs/extending.md. No download needed; data is generated on the fly.
_base_ = ['../_base_/default_runtime.py']

model = dict(
    type='mmmac.LSTMClassifier',
    in_channels=1,
    hidden_size=64,
    num_layers=1,
    num_classes=3,
)

train_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='mmmac.SyntheticWaveformDataset',
        num_samples=300,
        seq_len=64,
        noise=0.1,
        seed=0,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmmac.SyntheticWaveformDataset',
        num_samples=90,
        seq_len=64,
        noise=0.1,
        seed=1,
        test_mode=True,
    ),
)
val_evaluator = dict(type='mmmac.SimpleAccuracy')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# clip_grad is essential for RNNs: without it Adam can blow the recurrent
# weights up mid-training and accuracy collapses to chance level
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3),
    clip_grad=dict(max_norm=5.0),
)
param_scheduler = None

randomness = dict(seed=0)

# Plot every 10th val/test waveform with GT/pred labels to
# <work_dir>/<timestamp>/vis_data/vis_image/ (and wandb when enabled).
# Class names come from the dataset's metainfo automatically.
default_hooks = dict(
    visualization=dict(type='mmmac.WaveformVisualizationHook', enable=True,
                       interval=10))
