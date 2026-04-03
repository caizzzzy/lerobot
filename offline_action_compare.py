#!/usr/bin/env python

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import init_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline rollout-style action comparison on real robot dataset episodes."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to a train_config.json or a checkpoint/pretrained_model directory.",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Optional explicit policy checkpoint directory. Defaults to config-path directory.",
    )
    parser.add_argument("--dataset-repo-id", default=None, help="Override dataset repo_id.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root.")
    parser.add_argument(
        "--dataset-episodes",
        default=None,
        help="Optional JSON list of episode indices to evaluate, e.g. '[0,1,2]'.",
    )
    parser.add_argument(
        "--video-backend",
        default=None,
        choices=["pyav", "torchcodec", "video_reader"],
        help="Optional video decode backend override.",
    )
    parser.add_argument(
        "--shift-window",
        type=int,
        default=0,
        help="Optional temporal shift window. Compute best alignment over [-K, K]. Default 0.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only evaluate the first N selected episodes.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save full comparison report as JSON.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def maybe_parse_episodes(value):
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("--dataset-episodes must be a JSON list, e.g. '[0,1,2]'")
    return parsed


def resolve_policy_path(config_path: str, policy_path: str | None) -> Path:
    if policy_path is not None:
        return Path(policy_path)
    cfg_path = Path(config_path)
    return cfg_path if cfg_path.is_dir() else cfg_path.parent


def to_single_batch(item: dict) -> dict:
    batch = {}
    for key, value in item.items():
        if key == "action":
            continue
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def compute_shift_metrics(pred: np.ndarray, gt: np.ndarray, shift_window: int) -> dict:
    assert pred.shape == gt.shape
    result = {}

    diff = pred - gt
    result["strict_mse"] = float(np.mean(diff**2))
    result["strict_mae"] = float(np.mean(np.abs(diff)))
    result["per_dim_mse"] = np.mean(diff**2, axis=0).tolist()
    result["per_dim_mae"] = np.mean(np.abs(diff), axis=0).tolist()

    if shift_window <= 0:
        result["best_shift"] = 0
        result["best_shift_mse"] = result["strict_mse"]
        result["best_shift_mae"] = result["strict_mae"]
        return result

    best = None
    for shift in range(-shift_window, shift_window + 1):
        if shift < 0:
            pred_slice = pred[-shift:]
            gt_slice = gt[: len(pred_slice)]
        elif shift > 0:
            pred_slice = pred[:-shift]
            gt_slice = gt[shift:]
        else:
            pred_slice = pred
            gt_slice = gt

        if len(pred_slice) == 0:
            continue

        d = pred_slice - gt_slice
        mse = float(np.mean(d**2))
        mae = float(np.mean(np.abs(d)))
        if best is None or mse < best["best_shift_mse"]:
            best = {
                "best_shift": shift,
                "best_shift_mse": mse,
                "best_shift_mae": mae,
            }

    result.update(best or {"best_shift": 0, "best_shift_mse": result["strict_mse"], "best_shift_mae": result["strict_mae"]})
    return result


def main():
    args = parse_args()
    init_logging(console_level=args.log_level)

    cfg = TrainPipelineConfig.from_pretrained(args.config_path)
    cfg.policy.pretrained_path = resolve_policy_path(args.config_path, args.policy_path)

    if args.dataset_repo_id is not None:
        cfg.dataset.repo_id = args.dataset_repo_id
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.video_backend is not None:
        cfg.dataset.video_backend = args.video_backend
    cfg.dataset.episodes = maybe_parse_episodes(args.dataset_episodes)
    if args.device is not None:
        cfg.policy.device = args.device

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs = {"dataset_stats": dataset.meta.stats}
    postprocessor_kwargs = {}
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": cfg.policy.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    policy.eval()

    per_episode = []
    current_ep = None
    pred_actions = []
    gt_actions = []
    episodes_seen = 0

    def flush_episode(ep_idx):
        nonlocal pred_actions, gt_actions, per_episode, episodes_seen
        if ep_idx is None or len(pred_actions) == 0:
            return
        pred = np.stack(pred_actions)
        gt = np.stack(gt_actions)
        metrics = compute_shift_metrics(pred, gt, args.shift_window)
        metrics["episode_index"] = int(ep_idx)
        metrics["num_steps"] = int(len(pred))
        per_episode.append(metrics)
        episodes_seen += 1
        pred_actions = []
        gt_actions = []

    for i in range(len(dataset)):
        item = dataset[i]
        ep_idx = int(item["episode_index"].item())
        if current_ep is None or ep_idx != current_ep:
            flush_episode(current_ep)
            if args.max_episodes is not None and episodes_seen >= args.max_episodes:
                break
            current_ep = ep_idx
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            logging.info(f"Evaluating episode {ep_idx}")

        gt_action = item["action"].detach().cpu().numpy()
        observation = to_single_batch(item)
        observation = preprocessor(observation)
        with torch.no_grad():
            pred_action = policy.select_action(observation)
        pred_action = postprocessor(pred_action)
        pred_action = pred_action.squeeze(0).detach().cpu().numpy()

        pred_actions.append(pred_action)
        gt_actions.append(gt_action)

    flush_episode(current_ep)

    if not per_episode:
        raise ValueError("No episodes were evaluated.")

    overall = defaultdict(list)
    for ep in per_episode:
        for key in ["strict_mse", "strict_mae", "best_shift_mse", "best_shift_mae", "best_shift"]:
            overall[key].append(ep[key])

    report = {
        "policy_type": cfg.policy.type,
        "policy_path": str(cfg.policy.pretrained_path),
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": str(cfg.dataset.root) if cfg.dataset.root is not None else None,
        "dataset_episodes": cfg.dataset.episodes,
        "shift_window": args.shift_window,
        "overall": {
            "num_episodes": len(per_episode),
            "mean_strict_mse": float(np.mean(overall["strict_mse"])),
            "mean_strict_mae": float(np.mean(overall["strict_mae"])),
            "mean_best_shift_mse": float(np.mean(overall["best_shift_mse"])),
            "mean_best_shift_mae": float(np.mean(overall["best_shift_mae"])),
            "mean_best_shift": float(np.mean(overall["best_shift"])),
        },
        "per_episode": per_episode,
    }

    logging.info(json.dumps(report["overall"], indent=2, ensure_ascii=False))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved action comparison report to {output_path}")


if __name__ == "__main__":
    main()
