# CLAUDE.md — agent guide for this repository

OpenMMLab training/eval repo for macOS. Core code in `mmmac/`, experiments
in `configs/`, entry points in `tools/`, tests in `tests/` (mirrors
`mmmac/`). Docs in `docs/` are bilingual (EN + JA, same content twice) —
when editing docs, update BOTH language sections and keep mermaid diagrams
identical.

## Commands

```bash
./install.sh                          # full env setup (idempotent)
uv run pytest tests/unit -q           # fast tests
uv run pytest -q -m "not overfit"     # everything except the MNIST test
uv run pytest -q                      # full suite (downloads MNIST once)
uv run python tools/train.py <config> [--work-dir D] [--resume] [--cfg-options k=v ...]
uv run python tools/test.py <config> <checkpoint>
uv run python -m mmmac.utils.env      # environment / version report
```

Smoke config (no download, ~15 s):
`uv run python tools/train.py configs/time_series/lstm_waveform.py --cfg-options train_cfg.max_epochs=1`

## Hard rules

- **Never bump a single dependency version.** torch/mmcv/mmdet/mmseg/
  mmdet3d/mmpretrain are a pinned compatible set (see
  docs/installation.md). mmcv is compiled from source; the build needs
  `-Wno-invalid-specialization` (handled by install.sh) and
  `[tool.uv.extra-build-dependencies]` in pyproject.toml.
- **New code goes in `mmmac/`**, registered via decorators from
  `mmmac.registry` (never into upstream registries), exported from the
  sub-package `__init__.py`, referenced in configs as
  `type='mmmac.<Class>'`. Upstream components in configs always use scope
  prefixes: `type='mmpretrain.ClsHead'` etc.
- **Every `mmmac/<path>/<file>.py` gets `tests/unit/<path>/test_<file>.py`**
  (mirror structure). Runner-level behavior → `tests/integration/`;
  learning-capability checks → `tests/overfit/`.
- Configs inherit `configs/_base_/default_runtime.py`. Keep the gloo
  backend (macOS constraint). `num_workers=0` is the safe default;
  values > 0 work — pair with `persistent_workers=True`. wandb is opt-in
  via `configs/_base_/vis_wandb.py`.
- Device is auto-selected (MPS on Apple Silicon). `--device cpu|mps` on
  tools/train.py & test.py overrides; CPU is faster for tiny models
  (LeNet-size), MPS is ~17x faster for ResNet50-size models.
- `data/` and `work_dirs/` are gitignored scratch space — never commit
  them, never store code there.

## Extension patterns (working examples)

- Subclass an upstream model: `mmmac/models/backbones/lenet.py`
- Subclass an upstream dataset: `mmmac/datasets/mnist.py`
- From-scratch (time-series) model on mmengine BaseModel:
  `mmmac/models/temporal/lstm_classifier.py` +
  `mmmac/datasets/time_series.py` + `configs/time_series/lstm_waveform.py`
- Custom metric: `mmmac/evaluation/metrics/simple_accuracy.py`
- Custom hook (time-series visualization):
  `mmmac/engine/hooks/waveform_visualization_hook.py`

Details: docs/extending.md.
