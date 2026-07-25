#!/usr/bin/env python
"""Training entry point.

Examples:
    uv run python tools/train.py <config>
    uv run python tools/train.py <config> --work-dir work_dirs/my_run
    uv run python tools/train.py <config> --cfg-options train_cfg.max_epochs=10
"""
import argparse

from mmengine.config import Config, DictAction

from mmmac.apis import build_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a model from a config')
    parser.add_argument('config', help='path to the config file')
    parser.add_argument('--work-dir',
                        help='directory to save logs and checkpoints '
                        '(default: work_dirs/<config stem>)')
    parser.add_argument('--resume', nargs='?', const='auto', default=None,
                        help='resume training; without a value, resume from '
                        'the latest checkpoint in work_dir')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override config entries, e.g. '
                        'train_cfg.max_epochs=10 optim_wrapper.optimizer.lr=0.1')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    runner = build_runner(cfg, work_dir=args.work_dir,
                          cfg_options=args.cfg_options)
    runner.train()


if __name__ == '__main__':
    main()
