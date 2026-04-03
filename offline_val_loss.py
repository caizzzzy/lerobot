#!/usr/bin/env python

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute offline validation loss on a held-out LeRobot dataset split using a trained policy."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to a train_config.json or a checkpoint/pretrained_model directory containing train_config.json.",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Optional explicit path to the pretrained policy directory. "
        "If omitted, config-path directory is used as the pretrained path.",
    )
    parser.add_argument("--dataset-repo-id", default=None, help="Override validation dataset repo_id.")
    parser.add_argument("--dataset-root", default=None, help="Override validation dataset root.")
    parser.add_argument(
        "--dataset-episodes",
        default=None,
        help="Optional JSON list of episode indices to evaluate, e.g. '[0,1,2]'.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device, e.g. cuda, cuda:0, cpu. Defaults to config policy.device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size. Defaults to train config batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for validation. Default 0 for stability.",
    )
    parser.add_argument(
        "--video-backend",
        default=None,
        choices=["torchcodec", "pyav", "video_reader"],
        help="Optional override for dataset video backend.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after this many validation batches. Default: evaluate entire split.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level. Default: INFO",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the final metrics as JSON.",
    )
    return parser.parse_args()


def resolve_policy_path(config_path: str, policy_path: str | None) -> Path:
    if policy_path is not None:
        return Path(policy_path)

    cfg_path = Path(config_path)
    if cfg_path.is_dir():
        return cfg_path
    return cfg_path.parent


def maybe_parse_episodes(value: str | None):
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("--dataset-episodes must be a JSON list, e.g. '[0,1,2]'")
    return parsed


def build_dataloader(cfg: TrainPipelineConfig, dataset):
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    device = get_safe_torch_device(cfg.policy.device, log=False)
    return DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


def main():
    args = parse_args()
    init_logging(console_level=args.log_level)

    cfg = TrainPipelineConfig.from_pretrained(args.config_path)

    pretrained_path = resolve_policy_path(args.config_path, args.policy_path)
    cfg.policy.pretrained_path = pretrained_path

    if args.dataset_repo_id is not None:
        cfg.dataset.repo_id = args.dataset_repo_id
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.dataset_episodes is not None:
        cfg.dataset.episodes = maybe_parse_episodes(args.dataset_episodes)
    if args.video_backend is not None:
        cfg.dataset.video_backend = args.video_backend
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.policy.device = args.device

    logging.info(f"Loading validation dataset repo_id={cfg.dataset.repo_id} root={cfg.dataset.root}")
    dataset = make_dataset(cfg)
    logging.info(f"Validation dataset frames={dataset.num_frames} episodes={dataset.num_episodes}")

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs = {}
    postprocessor_kwargs = {}
    processor_kwargs["dataset_stats"] = dataset.meta.stats
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

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    dataloader = build_dataloader(cfg, dataset)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    policy.eval()

    total_weighted_loss = 0.0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="offline-val", leave=False)
        for batch in iterator:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Encountered non-finite validation loss at batch {batch_count}: {loss.item()}")

            bs = next(iter(batch.values())).shape[0]
            total_weighted_loss += float(loss.item()) * bs
            total_samples += bs
            batch_count += 1
            iterator.set_postfix(loss=f"{loss.item():.6f}")

            if args.max_batches is not None and batch_count >= args.max_batches:
                break

    if total_samples == 0:
        raise ValueError("No validation samples were processed.")

    avg_loss = total_weighted_loss / total_samples
    result = {
        "avg_val_loss": avg_loss,
        "num_batches": batch_count,
        "num_samples": total_samples,
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": str(cfg.dataset.root) if cfg.dataset.root is not None else None,
        "dataset_episodes": cfg.dataset.episodes,
        "policy_type": cfg.policy.type,
        "policy_path": str(cfg.policy.pretrained_path),
        "device": str(device),
    }

    logging.info(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved validation metrics to {out_path}")


if __name__ == "__main__":
    main()
