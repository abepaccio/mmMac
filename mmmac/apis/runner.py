"""Shared Runner construction used by tools/train.py, tools/test.py and the
tests. Keeps CLI entry points thin and gives tests a single API to drive
training programmatically.
"""
import copy
import os.path as osp
from contextlib import nullcontext
from typing import Optional
from unittest import mock

import mmengine.runner.runner as _runner_module
from mmengine.config import Config
from mmengine.runner import Runner


def build_runner(cfg: Config,
                 work_dir: Optional[str] = None,
                 cfg_options: Optional[dict] = None,
                 device: Optional[str] = None) -> Runner:
    """Build an mmengine Runner from a config.

    Priority for ``work_dir``: explicit argument > value in the config >
    ``work_dirs/<config filename stem>``.

    ``device`` overrides mmengine's automatic device selection ('mps' on
    Apple Silicon, else 'cpu'). Forcing 'cpu' is often faster for very
    small models, where MPS kernel-launch overhead dominates — see
    docs/training_and_testing.md.
    """
    cfg = copy.deepcopy(cfg)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir') is None:
        stem = 'default'
        if cfg.filename:
            stem = osp.splitext(osp.basename(cfg.filename))[0]
        cfg.work_dir = osp.join('work_dirs', stem)

    if cfg.get('launcher') is None:
        cfg.launcher = 'none'

    # The Runner moves the model to mmengine's get_device() during
    # construction; patching it here is the supported-surface way to pin
    # the device without forking the Runner.
    if device is not None:
        patch = mock.patch.object(_runner_module, 'get_device',
                                  return_value=device)
    else:
        patch = nullcontext()
    with patch:
        return Runner.from_cfg(cfg)
