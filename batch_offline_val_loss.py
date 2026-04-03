#!/usr/bin/env python

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple checkpoints in a training run and summarize checkpoint -> offline val loss."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing a checkpoints/ subdirectory.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        required=True,
        help="Validation dataset repo_id.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Parent root/path for validation dataset.",
    )
    parser.add_argument(
        "--dataset-episodes",
        default=None,
        help="Optional JSON list of held-out episode indices, e.g. '[80,81,82]'.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video decode backend used by offline validation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Validation dataloader workers. Default 0 for stability.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on batches per checkpoint for quick scans.",
    )
    parser.add_argument(
        "--step-start",
        type=int,
        default=None,
        help="Only evaluate checkpoints with step >= this value.",
    )
    parser.add_argument(
        "--step-end",
        type=int,
        default=None,
        help="Only evaluate checkpoints with step <= this value.",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Evaluate every Nth checkpoint after sorting by step. Default 1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of checkpoints to evaluate after filtering.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Evaluate checkpoints from latest to earliest.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path to save checkpoint -> val loss summary.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON path to save full summary.",
    )
    return parser.parse_args()


def discover_checkpoints(run_dir: Path):
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Expected checkpoints directory not found: {checkpoints_dir}")

    found = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        pretrained_dir = child / "pretrained_model"
        if pretrained_dir.is_dir():
            found.append((int(child.name), child, pretrained_dir))

    if not found:
        raise FileNotFoundError(f"No checkpoint directories with pretrained_model found in {checkpoints_dir}")

    found.sort(key=lambda x: x[0])
    return found


def filter_checkpoints(checkpoints, args):
    filtered = []
    for item in checkpoints:
        step = item[0]
        if args.step_start is not None and step < args.step_start:
            continue
        if args.step_end is not None and step > args.step_end:
            continue
        filtered.append(item)

    if args.reverse:
        filtered = list(reversed(filtered))

    if args.step_stride > 1:
        filtered = filtered[:: args.step_stride]

    if args.limit is not None:
        filtered = filtered[: args.limit]

    return filtered


def run_single_eval(pretrained_dir: Path, args):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        sys.executable,
        "offline_val_loss.py",
        "--config-path",
        str(pretrained_dir),
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--dataset-root",
        args.dataset_root,
        "--video-backend",
        args.video_backend,
        "--num-workers",
        str(args.num_workers),
        "--output-json",
        str(tmp_path),
    ]

    if args.dataset_episodes is not None:
        cmd.extend(["--dataset-episodes", args.dataset_episodes])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.max_batches is not None:
        cmd.extend(["--max-batches", str(args.max_batches)])

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(tmp_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
        result["status"] = "ok"
        return result
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "stdout": e.stdout,
            "stderr": e.stderr,
            "returncode": e.returncode,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    checkpoints = discover_checkpoints(run_dir)
    checkpoints = filter_checkpoints(checkpoints, args)

    if not checkpoints:
        raise ValueError("No checkpoints left after filtering.")

    rows = []
    print(f"Evaluating {len(checkpoints)} checkpoints from {run_dir}")

    for step, checkpoint_dir, pretrained_dir in checkpoints:
        print(f"[{step}] evaluating {pretrained_dir}")
        result = run_single_eval(pretrained_dir, args)
        row = {
            "step": step,
            "checkpoint_dir": str(checkpoint_dir),
            "pretrained_dir": str(pretrained_dir),
            "status": result["status"],
        }
        if result["status"] == "ok":
            row["avg_val_loss"] = result["avg_val_loss"]
            row["num_batches"] = result["num_batches"]
            row["num_samples"] = result["num_samples"]
            print(f"  avg_val_loss={result['avg_val_loss']:.6f} samples={result['num_samples']}")
        else:
            row["avg_val_loss"] = None
            row["num_batches"] = None
            row["num_samples"] = None
            row["returncode"] = result.get("returncode")
            row["stderr_tail"] = (result.get("stderr") or "")[-1000:]
            print(f"  FAILED returncode={row.get('returncode')}")

        rows.append(row)

    successful = [r for r in rows if r["status"] == "ok" and r["avg_val_loss"] is not None]
    if successful:
        best = min(successful, key=lambda r: r["avg_val_loss"])
        print("\nBest checkpoint:")
        print(f"  step={best['step']} avg_val_loss={best['avg_val_loss']:.6f}")
    else:
        print("\nNo checkpoint was evaluated successfully.")

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "step",
            "checkpoint_dir",
            "pretrained_dir",
            "status",
            "avg_val_loss",
            "num_batches",
            "num_samples",
            "returncode",
            "stderr_tail",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved CSV summary to {output_csv}")

    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON summary to {output_json}")


if __name__ == "__main__":
    main()
