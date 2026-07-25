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
