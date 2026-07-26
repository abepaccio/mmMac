# Opt-in wandb logging: add this file to `_base_` AFTER default_runtime.py,
# or copy the block below into an experiment config.
#
#   _base_ = ['../_base_/default_runtime.py', '../_base_/vis_wandb.py']
#
# Requires `wandb login` (or WANDB_API_KEY) once per machine.
# Offline mode: WANDB_MODE=offline uv run python tools/train.py ...
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(project='mmmac'),
    ),
]
visualizer = dict(type='Visualizer', vis_backends=vis_backends,
                  name='visualizer')
