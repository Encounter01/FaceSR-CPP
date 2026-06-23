# Maintainer Notes

## Public Documentation

Keep the README focused on:

- What the project does.
- Required dependencies.
- Build commands.
- Training and inference commands.
- Where configuration and documentation live.

Detailed implementation notes, evaluation scripts, and local analysis workflows should live in targeted documentation or script comments instead of the README.

## Runtime Assets

Datasets, generated outputs, and checkpoints are intentionally ignored by Git. Keep only placeholders such as `checkpoints/.gitkeep` in the repository.

## Model Compatibility

Native LibTorch checkpoints depend on the C++ model layout. When changing generator options such as attention, make sure inference commands and GUI behavior are documented clearly.

TorchScript models carry their own graph structure, but still need compatible preprocessing and postprocessing.

## Evaluation And Analysis Scripts

Scripts under `scripts/evaluation/`, `scripts/analysis/`, and `scripts/ablation/` are useful local tools. Avoid presenting generated local reports as canonical project documentation unless the required inputs, checkpoint names, script versions, and output paths are all reproducible.

## Release Checklist

- Build command-line targets.
- Run a small inference smoke test.
- Verify README command examples still match the CLI.
- Confirm large binary artifacts are not staged.
- Update `CHANGELOG.md`.
