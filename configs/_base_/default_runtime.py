# Shared runtime defaults for every experiment in this repository.
# Tuned for single-process training on macOS (no CUDA, no NCCL).
default_scope = 'mmmac'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),  # nccl is unavailable on macOS
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends,
                  name='visualizer')

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
