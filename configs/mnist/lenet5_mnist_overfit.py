# Overfit sanity check: memorize the first 64 MNIST training images.
# Validation runs on the SAME 64 samples, so top-1 accuracy must approach
# 100% — if it does not, the training loop / model / data path is broken.
# Used by tests/overfit/test_mnist_overfit.py.
_base_ = ['./lenet5_mnist.py']

num_overfit_samples = 64

train_dataloader = dict(
    batch_size=64,
    dataset=dict(indices=num_overfit_samples),
)

# validate memorization on the training subset itself
val_dataloader = dict(
    batch_size=64,
    dataset=dict(split='train', indices=num_overfit_samples),
)
test_dataloader = val_dataloader

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=50)

# constant LR + Adam converges fastest on a tiny subset
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='Adam', lr=1e-3))
param_scheduler = None

randomness = dict(seed=0, deterministic=False)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=200, max_keep_ckpts=1),
)
