#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only health check for a local LeRobot dataset. "
        "Find episodes that are most likely to crash training due to malformed data or video decode issues."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo_id / local dataset folder name")
    parser.add_argument("--root", required=True, help="Parent directory containing the dataset folder")
    parser.add_argument(
        "--backend",
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video decode backend used for spot checks. Default: pyav",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Tolerance passed to video frame decoding. Default: 1e-4",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only inspect the first N episodes. Default: inspect all episodes",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the full report as JSON",
    )
    return parser.parse_args()


def load_data_file_cached(cache: dict[Path, pd.DataFrame], fpath: Path) -> pd.DataFrame:
    if fpath not in cache:
        cache[fpath] = pd.read_parquet(fpath)
    return cache[fpath]


def add_issue(report: dict, episode_index: int, severity: str, message: str):
    report[episode_index]["issues"].append({"severity": severity, "message": message})
    report[episode_index]["severity_counts"][severity] += 1


def get_episode_rows(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    # Use the explicit `index` column rather than iloc so this still works if a parquet file contains
    # data for multiple episodes or chunks.
    ep_df = df[(df["index"] >= start_idx) & (df["index"] < end_idx)].copy()
    return ep_df.sort_values("index")


def inspect_episode(meta: LeRobotDatasetMetadata, ep_idx: int, backend: str, tolerance_s: float, cache: dict):
    ep = meta.episodes[ep_idx]
    result = {
        "episode_index": int(ep_idx),
        "issues": [],
        "severity_counts": defaultdict(int),
        "summary": {},
    }

    expected_len = int(ep["length"])
    start_idx = int(ep["dataset_from_index"])
    end_idx = int(ep["dataset_to_index"])

    data_file = meta.root / meta.get_data_file_path(ep_idx)
    if not data_file.exists():
        add_issue(result, ep_idx, "critical", f"Data parquet missing: {data_file}")
        return result

    try:
        df = load_data_file_cached(cache, data_file)
    except Exception as e:
        add_issue(result, ep_idx, "critical", f"Failed to read parquet {data_file}: {e}")
        return result

    ep_df = get_episode_rows(df, start_idx, end_idx)
    result["summary"]["row_count"] = int(len(ep_df))
    result["summary"]["expected_len"] = expected_len

    if len(ep_df) != expected_len:
        add_issue(
            result,
            ep_idx,
            "critical",
            f"Episode row count mismatch: expected {expected_len}, found {len(ep_df)}",
        )
        if len(ep_df) == 0:
            return result

    # Basic index checks.
    expected_indices = np.arange(start_idx, start_idx + len(ep_df))
    actual_indices = ep_df["index"].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        add_issue(result, ep_idx, "high", "Data `index` column is not contiguous within episode")

    expected_frame_indices = np.arange(len(ep_df))
    actual_frame_indices = ep_df["frame_index"].to_numpy()
    if not np.array_equal(actual_frame_indices, expected_frame_indices):
        add_issue(result, ep_idx, "high", "Data `frame_index` column is not contiguous from 0")

    episode_indices = ep_df["episode_index"].unique()
    if len(episode_indices) != 1 or int(episode_indices[0]) != ep_idx:
        add_issue(
            result,
            ep_idx,
            "high",
            f"`episode_index` column mismatch inside episode rows: {episode_indices.tolist()}",
        )

    timestamps = ep_df["timestamp"].to_numpy(dtype=np.float64)
    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        if not np.all(diffs > 0):
            add_issue(result, ep_idx, "high", "Timestamps are not strictly increasing")

        expected_ts = np.arange(len(timestamps), dtype=np.float64) / float(meta.fps)
        max_abs_err = float(np.max(np.abs(timestamps - expected_ts)))
        result["summary"]["max_timestamp_deviation_vs_uniform_s"] = max_abs_err
        if max_abs_err > (0.5 / float(meta.fps)):
            add_issue(
                result,
                ep_idx,
                "medium",
                f"Timestamps deviate from uniform {meta.fps}Hz grid by up to {max_abs_err:.6f}s",
            )

    # Video checks: existence + spot decode first/middle/last frame for each camera.
    for vid_key in meta.video_keys:
        try:
            video_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
        except Exception as e:
            add_issue(result, ep_idx, "critical", f"Failed to resolve video path for {vid_key}: {e}")
            continue

        if not video_path.exists():
            add_issue(result, ep_idx, "critical", f"Missing video file for {vid_key}: {video_path}")
            continue

        from_timestamp_key = f"videos/{vid_key}/from_timestamp"
        to_timestamp_key = f"videos/{vid_key}/to_timestamp"
        from_timestamp = float(ep[from_timestamp_key])
        to_timestamp = float(ep[to_timestamp_key])
        expected_video_span = len(ep_df) / float(meta.fps)
        if abs((to_timestamp - from_timestamp) - expected_video_span) > (0.5 / float(meta.fps)):
            add_issue(
                result,
                ep_idx,
                "medium",
                f"Video span mismatch for {vid_key}: "
                f"metadata span={to_timestamp - from_timestamp:.6f}s, expected~{expected_video_span:.6f}s",
            )

        if len(ep_df) == 0:
            continue

        sample_positions = sorted({0, len(ep_df) // 2, len(ep_df) - 1})
        sample_ts = [float(timestamps[i]) + from_timestamp for i in sample_positions]
        try:
            frames = decode_video_frames(video_path, sample_ts, tolerance_s=tolerance_s, backend=backend)
            if len(frames) != len(sample_ts):
                add_issue(
                    result,
                    ep_idx,
                    "high",
                    f"Decoded frame count mismatch for {vid_key}: expected {len(sample_ts)}, got {len(frames)}",
                )
        except Exception as e:
            add_issue(
                result,
                ep_idx,
                "critical",
                f"Video decode failed for {vid_key} using backend={backend}: {type(e).__name__}: {e}",
            )

    return result


def main():
    args = parse_args()
    root = Path(args.root)
    meta = LeRobotDatasetMetadata(args.repo_id, root=root / args.repo_id)

    data_cache: dict[Path, pd.DataFrame] = {}
    reports = []
    max_episodes = args.max_episodes if args.max_episodes is not None else len(meta.episodes)

    for ep_idx in range(min(max_episodes, len(meta.episodes))):
        reports.append(inspect_episode(meta, ep_idx, args.backend, args.tolerance_s, data_cache))

    # Rank by issue severity.
    severity_weight = {"critical": 100, "high": 10, "medium": 1}
    for report in reports:
        report["score"] = sum(
            report["severity_counts"][sev] * weight for sev, weight in severity_weight.items()
        )
        report["severity_counts"] = dict(report["severity_counts"])

    reports.sort(key=lambda x: (-x["score"], x["episode_index"]))

    total_flagged = sum(1 for r in reports if r["score"] > 0)
    print(f"Checked {len(reports)} episodes from {args.repo_id}")
    print(f"Flagged {total_flagged} episodes with at least one issue\n")

    for report in reports[:20]:
        if report["score"] == 0:
            break
        print(f"[episode {report['episode_index']}] score={report['score']} summary={report['summary']}")
        for issue in report["issues"]:
            print(f"  - {issue['severity']}: {issue['message']}")
        print()

    if total_flagged == 0:
        print("No obvious structural issues were found in the inspected episodes.")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        print(f"\nFull report written to: {output_path}")


if __name__ == "__main__":
    main()
