#!/usr/bin/env python3
"""Build and optionally run the quick ablation checkpoint evaluation command."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluation" / "final_recompute_eval.py"
EXPERIMENTS = [
    "a1_l1_only_50e",
    "a4_three_stage_50e",
    "a5_three_stage_attention_50e",
]
ATTENTION_EXPERIMENTS = {"a5_three_stage_attention_50e"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate quick FaceSR ablation checkpoints")
    parser.add_argument("--experiment", default="all", help="Experiment name or 'all'")
    parser.add_argument("--checkpoint-type", choices=["best", "latest", "both"], default="best")
    parser.add_argument("--n-images", type=int, default=3000)
    parser.add_argument("--device", choices=["auto", "cpu"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def selected_experiments(name: str) -> list[str]:
    if name == "all":
        return EXPERIMENTS
    if name not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment '{name}'. Valid values: all, {', '.join(EXPERIMENTS)}")
    return [name]


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def main() -> None:
    args = parse_args()
    suffixes = ["best", "latest"] if args.checkpoint_type == "both" else [args.checkpoint_type]
    specs: list[str] = []
    attention_labels: list[str] = []
    expected_paths: list[Path] = []

    for exp in selected_experiments(args.experiment):
        for suffix in suffixes:
            label = exp if len(suffixes) == 1 else f"{exp}_{suffix}"
            ckpt = REPO_ROOT / "checkpoints_ablation_quick" / exp / f"generator_{suffix}.pt"
            specs.append(f"{label}={rel(ckpt)}")
            expected_paths.append(ckpt)
            if exp in ATTENTION_EXPERIMENTS:
                attention_labels.append(label)

    output_dir = args.output_dir or REPO_ROOT / "results" / "ablation_quick_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--n-images",
        str(args.n_images),
        "--checkpoints",
        ",".join(specs),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
    ]
    if attention_labels:
        cmd.extend(["--attention-checkpoints", ",".join(attention_labels)])

    print("Evaluation command:")
    print(shlex.join(cmd))

    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        print("\nMissing checkpoint files:")
        for path in missing:
            print(f"- {rel(path)}")

    if not args.run:
        print("\nDry run only. Add --run to execute evaluation.")
        return

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
