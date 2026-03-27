# CLI Design — `python -m src`

## Overview

Add a Typer-based CLI to the project so that training and prediction can be invoked as:

```bash
uv run python -m src <command> <subcommand> [options]
```

No new entry points in `pyproject.toml`. No changes to existing module interfaces beyond extracting `train()` functions from the `if __name__ == "__main__"` guards.

---

## Files

### `src/__main__.py` (new)
Three-line entrypoint. Imports and runs the Typer app from `cli.py`.

```python
from src.cli import app

if __name__ == "__main__":
    app()
```

### `src/cli.py` (new)
All CLI logic lives here. Defines two Typer sub-apps (`train`, `predict`) composed into the root app.

---

## Commands

### `train unet`

```bash
uv run python -m src train unet [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 30 | Number of training epochs |
| `--lr` | float | 1e-4 | Learning rate |
| `--batch-size` | int | 8 | Batch size |
| `--image-size` | int | 256 | Input image size (256 or 512) |

### `train deeplabv3`

```bash
uv run python -m src train deeplabv3 [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 30 | Number of training epochs |
| `--lr` | float | 1e-4 | Learning rate |
| `--batch-size` | int | 8 | Batch size |

### `predict unet`

```bash
uv run python -m src predict unet --checkpoint <path> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | path | `outputs/checkpoints/best_unet.pth` | Path to `.pth` checkpoint file |
| `--split` | str | `valid` | Dataset split: `train`, `valid`, or `test` |
| `--threshold` | float | 0.4 | Binarization threshold for predictions |

### `predict deeplabv3`

```bash
uv run python -m src predict deeplabv3 [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | path | `outputs/checkpoints/best_deeplabv3.pth` | Path to `.pth` checkpoint file |
| `--split` | str | `valid` | Dataset split: `train`, `valid`, or `test` |

---

## Config Override Strategy

Both `train_unet.py` and `train_deeplabv3.py` have a `Config` class with hardcoded class attributes. Rather than refactoring them entirely, the CLI will:

1. Call a `make_config()` or equivalent function extracted from each train script
2. Override the relevant attributes with CLI-provided values
3. Pass the mutated config to a `train(config)` function

This requires extracting the training loop from the `if __name__ == "__main__"` block into a callable `train(config)` function in each script. The scripts remain runnable directly (`python src/training/train_unet.py`) with default config unchanged.

---

## Dependencies

Add `typer` to `pyproject.toml` dependencies:

```toml
"typer>=0.12.0",
```

---

## What is NOT in scope

- Baselines (`kmeans`, `rgb_thresholding`) — run directly as scripts
- Modal cloud training — invoked via `modal run scripts/modal_*.py`
- Preview utilities — run directly as scripts
- Config file support (YAML/JSON) — not needed for current use
