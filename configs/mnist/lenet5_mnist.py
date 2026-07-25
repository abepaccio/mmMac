# LeNet5 on MNIST — the reference image-classification recipe of this repo.
# All cross-library components use explicit scope prefixes (mmpretrain.*,
# mmmac.*) so the config is unambiguous about where each class comes from.
_base_ = ['../_base_/default_runtime.py']

model = dict(
    type='mmpretrain.ImageClassifier',
    data_preprocessor=dict(
        type='mmpretrain.ClsDataPreprocessor',
        # MNIST grayscale statistics
        mean=[33.46],
        std=[78.87],
    ),
    backbone=dict(type='mmmac.CustomLeNet5', num_classes=10),
    neck=None,
    head=dict(
        type='mmpretrain.ClsHead',
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
    ),
)

dataset_type = 'mmmac.MNIST'
data_root = 'data/mnist'
pipeline = [dict(type='mmpretrain.PackInputs')]

# num_workers=0 is the safe macOS default; >0 works too — pair it with
# persistent_workers=True (see docs/training_and_testing.md).
train_dataloader = dict(
    batch_size=128,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        pipeline=pipeline,
    ),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        pipeline=pipeline,
    ),
)
val_evaluator = dict(type='mmpretrain.Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4))
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[3],
                       gamma=0.1)
