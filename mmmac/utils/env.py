"""Environment report. Run ``uv run python -m mmmac.utils.env`` to print the
versions of every OpenMMLab package plus torch device availability — useful
when debugging installation issues on macOS.
"""
import importlib

import torch
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as mmengine_collect_env

from mmmac.version import __version__

_MM_PACKAGES = ('mmengine', 'mmcv', 'mmpretrain', 'mmdet', 'mmseg', 'mmdet3d')


def collect_env() -> dict:
    env = mmengine_collect_env()
    for pkg in _MM_PACKAGES:
        try:
            module = importlib.import_module(pkg)
            env[pkg] = getattr(module, '__version__', 'unknown')
        except ImportError as exc:
            env[pkg] = f'not available ({exc})'
    env['mmmac'] = f'{__version__}+{get_git_hash()[:7]}'
    env['MPS available'] = torch.backends.mps.is_available()
    return env


def main() -> None:
    for name, value in collect_env().items():
        print(f'{name}: {value}')


if __name__ == '__main__':
    main()
