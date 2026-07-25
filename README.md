# mmMac

OpenMMLab-based ML training / evaluation repository that runs natively on
macOS (Apple Silicon). mmengine, mmcv, mmpretrain, mmdetection,
mmsegmentation and mmdet3d are importable as modules and extendable from
the `mmmac/` package.

macOS (Apple Silicon) 上でネイティブに動作する、OpenMMLab ベースの
機械学習モデル学習・評価リポジトリ。mmengine / mmcv / mmpretrain /
mmdetection / mmsegmentation / mmdet3d をモジュールとして import し、
`mmmac/` パッケージから拡張して利用できます。

## Quick start / クイックスタート

```bash
./install.sh              # setup env / 環境構築

# tests / テスト
uv run pytest -q
```

## Documentation / ドキュメント

All docs are bilingual (English + 日本語) with mermaid diagrams —
start at [docs/README.md](docs/README.md).

| | |
| --- | --- |
| [docs/overview.md](docs/overview.md) | architecture & layout / アーキテクチャと構成 |
| [docs/installation.md](docs/installation.md) | setup & version pins / 環境構築とバージョン固定 |
| [docs/testing.md](docs/testing.md) | test suite / テスト構成 |

AI agents: see [CLAUDE.md](CLAUDE.md) for repo conventions and commands.
