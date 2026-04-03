# LeRobot Project Overview

## 1. Scope and version notes

This repository is based on Hugging Face `lerobot` and, in the checked-in `pyproject.toml`, corresponds to package version `0.4.4`.

For future work, the most important version-specific fact is this:

- The current upstream-style training stack in this workspace is **not Hydra-first**.
- It uses **`draccus` + Python dataclasses + nested CLI overrides** as the main configuration mechanism.
- The canonical config definitions live in `src/lerobot/configs/`, not in a Hydra YAML tree.

That matters because many older notes, blog posts, or informal summaries may still describe LeRobot as "Hydra-based". For this checkout, the practical mental model should be:

`CLI / --config_path JSON -> draccus parser -> TrainPipelineConfig dataclass -> train/eval scripts`

There is a second naming mismatch to keep in mind:

- Your prompt mentions `lerobot/common/{datasets,policies,envs}`.
- In this codebase revision, those responsibilities are split directly under `src/lerobot/`:
  - `src/lerobot/datasets/`
  - `src/lerobot/policies/`
  - `src/lerobot/envs/`
  - plus `src/lerobot/processor/`, `src/lerobot/optim/`, `src/lerobot/scripts/`

## 2. What LeRobot is doing at a high level

LeRobot is an end-to-end robot learning framework that standardizes four things:

1. Robot data representation
   - `LeRobotDataset v3.0` stores tabular signals in Parquet and visual streams in MP4, with metadata used to reconstruct episode-level views.
2. Policy definition
   - Policies such as ACT, Diffusion Policy, VQBeT, Pi0, SmolVLA, RTC, SARM, XVLA, etc. are implemented under `src/lerobot/policies/`.
3. Training / evaluation tooling
   - Main CLI scripts are registered in `pyproject.toml` as `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-replay`, `lerobot-teleoperate`, etc.
4. Hardware / environment abstraction
   - The same framework can talk to real robots, teleoperators, cameras, and simulation benchmarks.

The design goal is to make "dataset -> processor -> policy -> checkpoint -> evaluation / deployment" a repeatable pipeline.

## 3. Core training pipeline

### 3.1 Entry point

The core offline training entry is:

- `src/lerobot/scripts/lerobot_train.py`
- installed CLI alias: `lerobot-train`

The main function is `train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None)`.

### 3.2 End-to-end data flow

The training data flow is:

1. Parse config
   - `@parser.wrap()` in `src/lerobot/scripts/lerobot_train.py`
   - config class: `TrainPipelineConfig` in `src/lerobot/configs/train.py`

2. Validate / resolve policy config
   - `TrainPipelineConfig.validate()`:
     - loads policy from `--policy.path=...` if provided
     - or loads full training config from `--config_path=...`
     - auto-fills output dir, optimizer/scheduler presets, resume checkpoint path

3. Create `Accelerator`
   - `accelerate.Accelerator(...)` is used even in the normal training path
   - distributed training, gradient sync, and autocast are handled through Accelerate

4. Build dataset
   - `make_dataset(cfg)` in `src/lerobot/datasets/factory.py`
   - dataset metadata is read first through `LeRobotDatasetMetadata`
   - policy-defined temporal requirements are converted into `delta_timestamps`
   - actual dataset object is one of:
     - `LeRobotDataset`
     - `StreamingLeRobotDataset`

5. Build policy
   - `make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)`
   - the policy factory infers input/output feature specs from dataset metadata
   - if `policy.pretrained_path` exists, it loads config and weights from disk or HF Hub
   - otherwise, it instantiates a new policy from config

6. Build processors
   - `make_pre_post_processors(...)`
   - preprocessors normalize, rename, and move batch data into the policy-expected schema
   - postprocessors unnormalize policy outputs back into action space

7. Build optimizer and scheduler
   - `make_optimizer_and_scheduler(cfg, policy)`
   - by default, policies can inject their own optimizer/scheduler presets
   - you can disable that and provide explicit optimizer/scheduler config in the train config

8. Build sampler and DataLoader
   - plain shuffled `DataLoader`, or
   - `EpisodeAwareSampler` when the policy declares `drop_n_last_frames`

9. Training loop
   - `batch = next(dl_iter)`
   - `batch = preprocessor(batch)`
   - `loss, output_dict = policy.forward(batch)` inside `accelerator.autocast()`
   - `accelerator.backward(loss)`
   - grad clipping, optimizer step, scheduler step

10. Periodic side effects
   - log metrics
   - save checkpoints
   - run evaluation in sim if `cfg.env` is configured
   - optionally push final model + processors + train config to Hugging Face Hub

### 3.3 What the dataset object actually returns

`LeRobotDataset.__getitem__()` does more than "load one row":

- reads the current row from Parquet via Hugging Face `datasets`
- if temporal windows are requested:
  - computes extra frame/action indices inside the same episode
  - pads out-of-range positions with `*_is_pad`
- decodes aligned video frames from MP4 for visual keys
- applies image transforms if enabled
- appends a human-readable `task` string from metadata

So the policy usually receives a pre-assembled temporal training sample rather than a raw single timestamp.

### 3.4 Why policy config and dataset metadata are tightly coupled

The key bridge is `delta_timestamps`.

LeRobot does not hardcode one fixed temporal format for all policies. Instead:

- each policy config exposes temporal needs through properties such as:
  - `observation_delta_indices`
  - `action_delta_indices`
  - `reward_delta_indices`
- `make_dataset()` converts these index offsets into real timestamps using dataset FPS
- the dataset then loads exactly the temporal windows the policy expects

This is the main reason config debugging in LeRobot often boils down to:

- feature names matching
- tensor shapes matching
- temporal horizon / chunk settings matching
- dataset FPS and delta windows matching

## 4. LeRobotDataset v3.0: what to remember

For later environment and config work, the useful summary is:

- `meta/info.json`
  - schema, feature dtypes/shapes, FPS, path templates
- `meta/stats.json`
  - normalization stats used by processors and some policies
- `meta/tasks.parquet`
  - natural-language task mapping
- `meta/episodes/`
  - episode boundaries and offsets
- `data/`
  - Parquet shards with frame-aligned low-dimensional signals
- `videos/`
  - MP4 shards per camera stream

The important architectural idea is:

- storage is optimized file-wise
- user API stays episode/frame-wise
- metadata reconstructs episode boundaries and video offsets

This is why training can work equally from:

- a dataset pulled from the Hugging Face Hub
- a local cache under `root`
- streaming mode for very large datasets

## 5. Configuration system: current reality vs Hydra expectation

### 5.1 Canonical config code

The main config code is in:

- `src/lerobot/configs/train.py`
- `src/lerobot/configs/eval.py`
- `src/lerobot/configs/default.py`
- `src/lerobot/configs/policies.py`
- `src/lerobot/configs/parser.py`

This is the config layer you should read first when debugging training parameters.

### 5.2 What replaced Hydra here

Current LeRobot behavior is:

- config schema = Python dataclasses
- parser = `draccus`
- override syntax = nested CLI flags that look Hydra-like

Examples:

```bash
--dataset.repo_id=lerobot/pusht
--policy.type=act
--batch_size=64
--steps=200000
--optimizer.lr=1e-4
--scheduler.num_warmup_steps=500
--policy.device=cuda
```

So the **user experience resembles Hydra-style nested overrides**, but the implementation is not Hydra.

### 5.3 Three important config loading modes

#### Mode A: train from explicit CLI fields

Recommended mental model:

```bash
lerobot-train \
  --dataset.repo_id=<dataset> \
  --policy.type=<policy_type> \
  ...
```

#### Mode B: train from a saved JSON config

```bash
lerobot-train \
  --config_path=/path/to/train_config.json
```

`parser.wrap()` detects `--config_path` and loads the dataclass from JSON.

#### Mode C: start from an existing pretrained policy

```bash
lerobot-train \
  --dataset.repo_id=<dataset> \
  --policy.path=/path/to/pretrained_model_or_hf_repo \
  ...
```

This lets training resolve the policy config from the saved `config.json`.

### 5.4 Resume training

Resume is checkpoint-centric:

```bash
lerobot-train \
  --config_path=outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

That works because checkpoints store:

- policy weights and `config.json`
- `train_config.json`
- processor config/state
- optimizer/scheduler/RNG state

### 5.5 The top-level `configs/` directory in this workspace

The root-level `configs/*.json` files are **not** the upstream canonical config system.

They are best understood as:

- local, concrete experiment snapshots
- directly loadable through `--config_path=...`
- useful templates for this workspace's custom datasets and robot setups

Examples observed in this repo:

- `configs/train_config.json`
- `configs/train_config_act.json`
- `configs/train_config_tcp.json`
- `configs/train_config_tcp6d.json`
- `configs/train_config_umi.json`

These files are especially useful because they show real values for:

- `input_features` / `output_features`
- local dataset roots under `data/`
- image transform settings
- diffusion / ACT hyperparameters
- WandB settings
- local output directory conventions

### 5.6 One important CLI gotcha

In current code, `PreTrainedConfig.push_to_hub` defaults to `true`.

That means a training run usually needs one of:

- `--policy.repo_id=<hf_user>/<model_repo>`
- or `--policy.push_to_hub=false`

This is stricter than some older minimal examples in docs/README.

## 6. Module and script map

### 6.1 Core packages under `src/lerobot/`

- `datasets/`
  - dataset metadata, Parquet/MP4 loading, streaming, transforms, stats, dataset editing utilities
- `policies/`
  - policy configs, model implementations, policy-specific processors
- `envs/`
  - benchmark/simulation config and creation logic
- `processor/`
  - preprocessing and postprocessing pipelines that bridge raw data and policy schema
- `optim/`
  - optimizer and scheduler config/build logic
- `scripts/`
  - CLI entry implementations
- `robots/`, `teleoperators/`, `cameras/`
  - real-world hardware abstraction and data collection stack
- `async_inference/`
  - policy server / robot client for decoupled inference
- `utils/`
  - logging, random state, checkpoint helpers, import utilities, etc.

### 6.2 Most relevant CLI scripts

- `lerobot-train`
  - offline training main loop
- `lerobot-eval`
  - rollout-based evaluation in simulation / supported envs
- `lerobot-record`
  - collect dataset from real robot or policy-assisted control
- `lerobot-replay`
  - replay saved actions to hardware
- `lerobot-teleoperate`
  - manual teleoperation entry
- `lerobot-dataset-viz`
  - inspect dataset content
- `lerobot-edit-dataset`
  - dataset manipulation
- `lerobot-train-tokenizer`
  - tokenizer training for policies that need it

### 6.3 Evaluation path

`lerobot-eval` loads:

- an env via `make_env`
- env-specific processors if needed
- a policy via `policy.path`
- policy pre/postprocessors

Then it runs vectorized rollouts and aggregates:

- reward
- success rate
- optional videos

This is a separate path from checkpoint-time evaluation inside `lerobot-train`, but it reuses the same processor/policy conventions.

## 7. Hardware, environment, CUDA, multi-GPU

### 7.1 Base environment requirements

From `pyproject.toml`, the core baseline is:

- Python `>=3.10`
- `torch>=2.2.1,<2.8.0`
- `accelerate>=1.10.0,<2.0.0`
- `datasets`, `diffusers`, `huggingface-hub`
- `torchvision`, `draccus`, `gymnasium`, `wandb`

The installation docs recommend:

- Conda / Miniforge
- Python 3.10 env
- `ffmpeg` installed inside the Conda environment

### 7.2 Accelerator support

LeRobot supports:

- CPU
- CUDA
- MPS
- XPU

Device selection rules:

- if `policy.device` is invalid or omitted, config code auto-selects the best available device
- in training, `Accelerator` ultimately controls the runtime device placement
- if `policy.device=cpu`, training explicitly forces CPU mode

### 7.3 Mixed precision

There are two layers to remember:

1. `policy.use_amp`
   - config-level flag
   - mainly relevant for non-Accelerate inference/eval/control paths

2. `accelerate` mixed precision
   - main training path uses `accelerator.autocast()`
   - in multi-GPU training, precision is controlled by Accelerate launch/config, not just by `policy.use_amp`

### 7.4 Multi-GPU

Multi-GPU is handled through Hugging Face Accelerate, not custom DDP wiring.

Typical launch pattern:

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  $(which lerobot-train) \
  --dataset.repo_id=<dataset> \
  --policy.type=act \
  --policy.repo_id=<model_repo>
```

Important operational facts:

- only main process logs and saves checkpoints
- effective batch size = `batch_size * num_gpus`
- LeRobot does **not** auto-scale learning rate or training steps for you
- scheduler stepping is configured to avoid incorrect process-count scaling

### 7.5 Policy- or hardware-specific extras

Many capabilities are optional extras:

- simulation: `aloha`, `pusht`, `libero`, `metaworld`
- hardware: `feetech`, `dynamixel`, `intelrealsense`, `reachy2`, `unitree_g1`, etc.
- advanced policies: `smolvla`, `groot`, `xvla`, `sarm`, `peft`, `wallx`

For later Conda/environment work, check `pyproject.toml` first before assuming one environment fits all policies.

## 8. Local workspace-specific observations

This workspace is not a pristine upstream checkout. It also contains local project artifacts:

- root-level experiment configs in `configs/`
- local datasets under `data/`
- local outputs under `outputs/`
- `diana_sdk/` and custom robot integration code
- extra docs in `docs/PROJECT_OVERVIEW.md`, `docs/REAL_WORLD_ACT_DIFFUSION_PLAN.md`, etc.

For later work, the safest separation is:

- upstream LeRobot core logic: `src/lerobot/` + `docs/source/`
- workspace-local operational choices: root `configs/`, local robot scripts, local data/output directories

## 9. The training control chain you will most likely touch later

When debugging configs, these files are the shortest path to truth:

1. `src/lerobot/scripts/lerobot_train.py`
   - actual train loop and checkpoint/eval cadence
2. `src/lerobot/configs/train.py`
   - what fields exist, what is required, what gets auto-filled
3. `src/lerobot/configs/policies.py`
   - common policy fields like device, Hub behavior, input/output features
4. `src/lerobot/policies/<policy>/configuration_*.py`
   - policy-specific hyperparameters and temporal windows
5. `src/lerobot/datasets/factory.py`
   - how dataset windows are derived from policy config
6. `src/lerobot/datasets/lerobot_dataset.py`
   - what the batch payload really looks like
7. `src/lerobot/policies/factory.py`
   - how dataset metadata becomes policy input/output schema
8. `src/lerobot/optim/`
   - optimizer/scheduler defaults and override behavior

## 10. Recommended command templates

### 10.1 Train from scratch with explicit CLI overrides

Recommended explicit format:

```bash
lerobot-train \
  --dataset.repo_id=<dataset_repo_or_local_name> \
  --policy.type=<act|diffusion|vqbet|pi0|smolvla|...> \
  --policy.repo_id=<hf_user>/<model_repo> \
  --output_dir=outputs/train/<run_name> \
  --job_name=<run_name> \
  --policy.device=cuda
```

If you do not want Hub upload at the end, replace `--policy.repo_id=...` with:

```bash
--policy.push_to_hub=false
```

### 10.2 Train from a saved JSON config

```bash
lerobot-train \
  --config_path=configs/train_config.json
```

You can still override individual fields on top:

```bash
lerobot-train \
  --config_path=configs/train_config.json \
  --batch_size=32 \
  --steps=50000 \
  --optimizer.lr=5e-5
```

### 10.3 Resume from a checkpoint

```bash
lerobot-train \
  --config_path=outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

### 10.4 Continue training from an existing pretrained policy

```bash
lerobot-train \
  --dataset.repo_id=<dataset> \
  --policy.path=<local_checkpoint_dir_or_hf_model_repo> \
  --policy.push_to_hub=false
```

### 10.5 Multi-GPU

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=<N> \
  $(which lerobot-train) \
  --dataset.repo_id=<dataset> \
  --policy.type=<policy> \
  --policy.repo_id=<hf_user>/<model_repo>
```

## 11. Bottom-line summary for future debugging

If you only remember a few facts, remember these:

- Current LeRobot config logic in this repo is **Draccus/dataclass-based**, not Hydra-YAML-based.
- `lerobot-train` is the main training controller.
- `TrainPipelineConfig` and policy config classes are the source of truth for allowed parameters.
- Dataset temporal windows are derived from policy config, then materialized by `LeRobotDataset`.
- Processors are first-class components: a lot of "shape mismatch" and "normalization mismatch" bugs live there.
- Multi-GPU and mixed precision are mediated by `accelerate`.
- Root `configs/*.json` are local experiment snapshots, not the upstream canonical config registry.

## 12. References

### Workspace code and docs

- `README.md`
- `pyproject.toml`
- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/scripts/lerobot_eval.py`
- `src/lerobot/configs/`
- `src/lerobot/datasets/`
- `src/lerobot/policies/`
- `src/lerobot/envs/`
- `docs/source/installation.mdx`
- `docs/source/lerobot-dataset-v3.mdx`
- `docs/source/multi_gpu_training.mdx`
- `docs/source/bring_your_own_policies.mdx`
- `docs/source/torch_accelerators.mdx`
- `docs/source/il_robots.mdx`

### Official Hugging Face docs referenced for alignment

- https://huggingface.co/docs/lerobot/installation
- https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- https://huggingface.co/docs/lerobot/multi_gpu_training
- https://huggingface.co/docs/lerobot/bring_your_own_policies
- https://huggingface.co/docs/lerobot/torch_accelerators
- https://huggingface.co/docs/lerobot/il_robots
