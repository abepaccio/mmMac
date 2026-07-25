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
uv run python -m mmmac.utils.env      # environment / version report
```

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
  backend (macOS constraint).
- `data/` and `work_dirs/` are gitignored scratch space — never commit
  them, never store code there.

## Extension patterns (working examples)

(none yet — see docs/extending.md as patterns are added)
