"""Smoke tests for the CLI entry points (tools/train.py, tools/test.py)."""
import subprocess
import sys

import pytest


def _run(repo_root, *argv):
    return subprocess.run([sys.executable, *argv], cwd=repo_root,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.integration
def test_train_help(repo_root):
    result = _run(repo_root, 'tools/train.py', '--help')
    assert result.returncode == 0
    assert '--cfg-options' in result.stdout


@pytest.mark.integration
def test_test_help(repo_root):
    result = _run(repo_root, 'tools/test.py', '--help')
    assert result.returncode == 0
    assert 'checkpoint' in result.stdout


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_train_then_test_via_cli(tmp_path, repo_root):
    work_dir = tmp_path / 'run'
    train = _run(
        repo_root, 'tools/train.py', 'configs/time_series/lstm_waveform.py',
        '--work-dir', str(work_dir),
        '--cfg-options', 'train_cfg.max_epochs=1',
        'train_dataloader.dataset.num_samples=60',
        'val_dataloader.dataset.num_samples=30',
        'test_dataloader.dataset.num_samples=30',
    )
    assert train.returncode == 0, train.stderr[-2000:]
    checkpoint = work_dir / 'epoch_1.pth'
    assert checkpoint.exists()

    test = _run(
        repo_root, 'tools/test.py', 'configs/time_series/lstm_waveform.py',
        str(checkpoint),
        '--work-dir', str(tmp_path / 'eval'),
        '--cfg-options', 'test_dataloader.dataset.num_samples=30',
    )
    assert test.returncode == 0, test.stderr[-2000:]
    assert 'accuracy/top1' in (test.stdout + test.stderr)
