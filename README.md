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
./install.sh --with-data   # setup env (+ download MNIST) / 環境構築(+MNIST取得)

# overfit sanity check — reaches 100% top-1 in ~1 min
# 過学習サニティチェック — 約1分で top-1 100% に到達
uv run python tools/train.py configs/mnist/lenet5_mnist_overfit.py

# full MNIST training / MNIST 本学習
uv run python tools/train.py configs/mnist/lenet5_mnist.py

# evaluate a checkpoint / チェックポイント評価
uv run python tools/test.py configs/mnist/lenet5_mnist.py \
    work_dirs/lenet5_mnist/epoch_5.pth

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
| [docs/training_and_testing.md](docs/training_and_testing.md) | train/test CLI / 学習・評価CLI |
| [docs/extending.md](docs/extending.md) | custom models / モデル拡張 |
| [docs/testing.md](docs/testing.md) | test suite / テスト構成 |

AI agents: see [CLAUDE.md](CLAUDE.md) for repo conventions and commands.
