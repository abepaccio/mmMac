"""Shared Runner construction used by tools/train.py, tools/test.py and the
tests. Keeps CLI entry points thin and gives tests a single API to drive
training programmatically.
"""
import copy
import os.path as osp
from typing import Optional

from mmengine.config import Config
from mmengine.runner import Runner


def build_runner(cfg: Config,
                 work_dir: Optional[str] = None,
                 cfg_options: Optional[dict] = None) -> Runner:
    """Build an mmengine Runner from a config.

    Priority for ``work_dir``: explicit argument > value in the config >
    ``work_dirs/<config filename stem>``.
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

    return Runner.from_cfg(cfg)
