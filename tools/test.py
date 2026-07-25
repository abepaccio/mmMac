#!/usr/bin/env python
"""Evaluation entry point.

Example:
    uv run python tools/test.py <config> <checkpoint>
"""
import argparse

from mmengine.config import Config, DictAction

from mmmac.apis import build_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a checkpoint with the test dataloader')
    parser.add_argument('config', help='path to the config file')
    parser.add_argument('checkpoint', help='path to the checkpoint file')
    parser.add_argument('--work-dir',
                        help='directory to save evaluation logs '
                        '(default: work_dirs/<config stem>)')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'],
                        default=None,
                        help='force the compute device (default: auto — mps '
                        'on Apple Silicon)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override config entries, e.g. '
                        'test_dataloader.batch_size=256')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint

    runner = build_runner(cfg, work_dir=args.work_dir,
                          cfg_options=args.cfg_options, device=args.device)
    runner.test()


if __name__ == '__main__':
    main()
