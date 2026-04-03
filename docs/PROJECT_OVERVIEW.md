# PROJECT_OVERVIEW

## 1. 项目总体目标

### 1.1 这个仓库主要在做什么

- 已确认：仓库主干是 Hugging Face `lerobot`，目标是提供真实机器人场景下的数据集格式、训练、评测、采集、回放、远程推理与硬件接入能力，核心定位见 [README.md](../README.md)、[pyproject.toml](../pyproject.toml)、[src/lerobot/scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py)。
- 已确认：当前仓库不是纯上游镜像，已经混入一批本地项目定制内容；这些文件“存在并可运行”是已确认的，但“是否已经成为仓库内最合理/最推荐方案”不能仅凭存在本身下结论。重点集中在：
  - Diana 机械臂接入：[src/lerobot/robots/diana_follower/](../src/lerobot/robots/diana_follower/)
  - 根目录自定义推理入口：[run_inference.sh](../run_inference.sh)、[diffusion_using_diana.py](../diffusion_using_diana.py)
  - 一批面向本地 HDF5 数据的转换脚本：[convert_to_lerobot*.py](../)
  - 本地训练配置：[configs/*.json](../configs/)

### 1.2 更偏训练 / 推理 / 数据集转换 / 机器人控制 / 评测 / 工具链中的哪些部分

| 方向 | 结论 | 证据 |
| --- | --- | --- |
| 离线训练 | 已确认：强 | [src/lerobot/scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py), [configs/train_config*.json](../configs/) |
| 真实机器人推理 | 已确认：仓库内同时存在官方/上游真实机器人推理思路与当前项目自定义推理脚本，但优先级不能只凭文件存在直接定性 | [run_inference.sh](../run_inference.sh), [diffusion_using_diana.py](../diffusion_using_diana.py), [src/lerobot/async_inference/](../src/lerobot/async_inference/), [examples/rtc/eval_with_real_robot.py](../examples/rtc/eval_with_real_robot.py), [docs/source/async.mdx](../docs/source/async.mdx), [src/lerobot/scripts/lerobot_eval.py](../src/lerobot/scripts/lerobot_eval.py) |
| 数据集转换 | 已确认：很强，且高度本地化 | [convert_to_lerobot*.py](../), [check_data_accuracy.py](../check_data_accuracy.py) |
| 机器人控制 / 采集 | 已确认：强 | [src/lerobot/scripts/lerobot_record.py](../src/lerobot/scripts/lerobot_record.py), [src/lerobot/scripts/lerobot_teleoperate.py](../src/lerobot/scripts/lerobot_teleoperate.py), [src/lerobot/scripts/lerobot_replay.py](../src/lerobot/scripts/lerobot_replay.py) |
| 仿真评测 | 已确认：存在，但不是当前仓库最核心的本地改造方向 | [src/lerobot/scripts/lerobot_eval.py](../src/lerobot/scripts/lerobot_eval.py), [examples/rtc/](../examples/rtc/) |
| RL / VLA 扩展 | 已确认：上游能力完整保留 | [src/lerobot/rl/](../src/lerobot/rl/), [src/lerobot/policies/](../src/lerobot/policies/) |

### 1.3 与真实机器人、相机、机械臂、数据集之间的关系

- 已确认：`Robot` 抽象层要求每个机器人实现 `get_observation()` / `send_action()` / `observation_features` / `action_features`，见 [src/lerobot/robots/robot.py](../src/lerobot/robots/robot.py)。
- 已确认：数据集标准格式是 LeRobotDataset v3.0，主数据在 Parquet，视觉模态在 MP4 或 image，见 [README.md](../README.md)、[src/lerobot/datasets/lerobot_dataset.py](../src/lerobot/datasets/lerobot_dataset.py)、[src/lerobot/datasets/utils.py](../src/lerobot/datasets/utils.py)。
- 已确认：当前仓库里唯一明显的本地新增真实机器人适配重点是 Diana 机械臂，摄像头与夹爪状态通过 ROS2 接入，机械臂本体通过 Diana SDK 接入，见 [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py)、[diana_sdk/README.md](../diana_sdk/README.md)。
- 高概率推测：本地业务链路是“采集 AgileX/UMI 原始数据 -> 转为 LeRobotDataset -> 训练 diffusion/ACT -> 在 Diana 上推理执行”。依据是转换脚本命名、训练配置、推理入口和输出路径相互对应，但没有一份统一 README 把这条链路完整写死。

## 2. 项目主干结构

### 2.1 根目录主要目录作用

| 目录 | 作用 | 判断 |
| --- | --- | --- |
| `src/lerobot/` | 核心代码，通用框架、模型、数据集、机器人、相机、处理流水线 | 已确认 |
| `examples/` | 官方/上游示例，展示训练、数据集、RTC、不同机器人用法 | 已确认 |
| `configs/` | 当前项目的本地训练配置，明显不是上游通用配置 | 已确认 |
| `tests/` | 上游主干测试集，覆盖 dataset/processor/policy/async/camera/robot 等模块 | 已确认 |
| `diana_sdk/` | Diana 机械臂 Python SDK 与动态库 | 已确认 |
| `docs/source/` | 上游文档源文件 | 已确认 |
| `benchmarks/` | 基准与辅助工具，当前项目优先级较低 | 已确认 |
| `outputs/` | 训练输出与 checkpoint，非源码 | 已确认 |
| `data/` | 本地数据集产物目录，非源码 | 已确认 |
| `media/`, `picture/` | 资源或调试图片目录 | 已确认 |
| `pytorch3d/` | 三方源码/依赖副本，不是本项目主干 | 高概率推测 |

### 2.2 根目录重要文件分类

| 文件/模式 | 作用 | 分类 |
| --- | --- | --- |
| [pyproject.toml](../pyproject.toml) | 依赖、可选 extra、CLI 注册 | 核心 |
| [setup.py](../setup.py) | 打包补充逻辑 | 核心但简单 |
| [README.md](../README.md) | 上游项目说明 | 背景说明 |
| [run_inference.sh](../run_inference.sh) | 当前项目自定义的异步推理部署脚本 | 核心本地入口 |
| [diffusion_using_diana.py](../diffusion_using_diana.py) | 当前项目自定义的同步式 Diana 推理脚本 | 核心本地入口 |
| [convert_to_lerobot*.py](../) | 当前项目自定义的数据转换脚本族 | 核心本地工具 |
| [check_data_accuracy.py](../check_data_accuracy.py) | 临时数据列检查 | 调试脚本 |
| [train_lerobot.slurm](../train_lerobot.slurm) | 集群训练提交脚本 | 辅助脚本 |
| `test_*.py`, [look_parquet.py](../look_parquet.py) | 根目录临时验证/探索脚本 | 实验性/调试性 |

### 2.3 哪些目录是核心，哪些更像辅助或实验

- 核心代码：`src/lerobot/`, `configs/`, 根目录定制入口脚本。
- 辅助内容：`examples/`, `docs/source/`, `benchmarks/`, `train_lerobot.slurm`。
- 实验/临时堆积：根目录多个 `convert_to_lerobot*.py`、`test_*.py`、`look_parquet.py`、`check_data_accuracy.py`、`picture/`。

## 3. 核心入口

### 3.1 真正常用入口总览

| 场景 | 推荐入口 | 典型用途 | 调用链 |
| --- | --- | --- | --- |
| 训练 | [src/lerobot/scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py) 或 CLI `lerobot-train` | 离线训练 policy | `TrainPipelineConfig` -> `make_dataset` -> `make_policy` -> `make_pre_post_processors` |
| 推理（同步，本地直连） | [diffusion_using_diana.py](../diffusion_using_diana.py) | 直接在本机加载模型并控制 Diana | `DiffusionPolicy.from_pretrained` -> `build_inference_frame` -> `select_action` -> `DianaFollower.send_action` |
| 推理（异步，自定义链路） | [run_inference.sh](../run_inference.sh) | 启动远程策略服务端 + 本地机器人客户端 | `policy_server` + `robot_client` gRPC 协同 |
| 机器人采集 | [src/lerobot/scripts/lerobot_record.py](../src/lerobot/scripts/lerobot_record.py) | 采集真实机器人数据集 | `make_robot_from_config` + `teleop/policy` + `LeRobotDataset` |
| 机器人遥操作 | [src/lerobot/scripts/lerobot_teleoperate.py](../src/lerobot/scripts/lerobot_teleoperate.py) | 直接 teleop 控制 | `make_teleoperator_from_config` -> `robot.send_action` |
| 数据回放 | [src/lerobot/scripts/lerobot_replay.py](../src/lerobot/scripts/lerobot_replay.py) | 按数据集 action 重放到机器人 | `LeRobotDataset` -> `robot.send_action` |
| 数据转换 | [convert_to_lerobot*.py](../) | HDF5/本地原始数据转 LeRobotDataset | 脚本内自建 `meta/`, `data/`, `videos/` |
| 数据检查/可视化 | [src/lerobot/scripts/lerobot_dataset_viz.py](../src/lerobot/scripts/lerobot_dataset_viz.py), [check_data_accuracy.py](../check_data_accuracy.py), [look_parquet.py](../look_parquet.py) | 看 parquet/video/列结构 | 直接读 dataset/parquet |

### 3.2 训练入口

- 已确认：标准训练入口是 CLI `lerobot-train`，注册于 [pyproject.toml](../pyproject.toml)，实现见 [src/lerobot/scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py)。
- 高概率推测：当前项目实际训练时大概率使用 `--config_path=configs/train_config*.json`，因为根目录已有多份本地配置和 [train_lerobot.slurm](../train_lerobot.slurm)；但这只能说明“你本地常用/曾用过”，不能说明“这是仓库层面的最佳实践”。
- 用户通常要改：
  - `dataset.root` / `dataset.repo_id`
  - `policy.type`
  - `policy.input_features` / `output_features`
  - `output_dir`
  - `device`, `batch_size`, `steps`, `num_workers`
  - 图像增强和视频后端

### 3.3 推理入口

#### A. [run_inference.sh](../run_inference.sh)

- 已确认：这是当前仓库中的一个自定义异步推理部署脚本。
- 高概率推测：它很接近你本地曾经的真实使用方式，因为它把 `policy_server`、`robot_client`、ROS2 环境、Diana 配置、checkpoint 路径都串了起来。
- 未确认：它是否应被视为“当前仓库最贴近真实部署的入口”。仓库内同时还存在：
  - 官方 async 文档与示例：[docs/source/async.mdx](../docs/source/async.mdx)、[examples/tutorial/async-inf/](../examples/tutorial/async-inf/)
  - RTC 真机示例：[examples/rtc/eval_with_real_robot.py](../examples/rtc/eval_with_real_robot.py)、[docs/source/rtc.mdx](../docs/source/rtc.mdx)
  - 官方统一评测入口：[src/lerobot/scripts/lerobot_eval.py](../src/lerobot/scripts/lerobot_eval.py)
- 用途：
  - `server` 模式启动策略服务器：`python -m lerobot.async_inference.policy_server`
  - `client` 模式启动机器人客户端：`python -m lerobot.async_inference.robot_client`
- 典型参数：
  - `ROBOT_TYPE=diana_follower`
  - `ROBOT_PORT=<机器人 IP>`
  - `CAMERAS=<相机 JSON>`
  - `POLICY_TYPE=diffusion`
  - `MODEL_PATH=<checkpoint/pretrained_model>`
  - `POLICY_DEVICE=cuda`
  - `ACTIONS_PER_CHUNK`
  - gRPC 地址和聚合策略
- 已确认：脚本依赖 ROS2 Humble 和本地 `pika_ros` 环境，见脚本顶部环境变量。
- 高概率推测：这个脚本更像“为当前本地部署环境定制的启动器”，而不是可直接推广给所有仓库使用者的通用入口。

#### B. [diffusion_using_diana.py](../diffusion_using_diana.py)

- 已确认：这是更直接的单进程推理脚本。
- 已确认：它同样是当前项目自定义脚本，而不是上游标准 CLI。
- 用途：本地加载 `DiffusionPolicy`，直接从 `DianaFollower` 读观测并发送动作。
- 调用链：
  - `DiffusionPolicy.from_pretrained`
  - `LeRobotDatasetMetadata`
  - `make_pre_post_processors`
  - `build_inference_frame`
  - `model.select_action`
  - `make_robot_action`
  - `adapt_action_for_diana`
  - `robot.send_action`
- 用户通常要改：
  - `model_id`
  - `dataset_id`
  - `follower_port`
  - `robot_cfg` 内 ROS topic / cameras / control_mode
  - `device`

### 3.4 机器人控制或部署入口

- 已确认：通用控制入口是 [src/lerobot/scripts/lerobot_teleoperate.py](../src/lerobot/scripts/lerobot_teleoperate.py)。
- 已确认：仓库内与“真实机器人部署/执行”相关的候选入口至少有四类：
  - 自定义同步脚本：[diffusion_using_diana.py](../diffusion_using_diana.py)
  - 自定义异步脚本：[run_inference.sh](../run_inference.sh)
  - 官方 async 机制：[src/lerobot/async_inference/](../src/lerobot/async_inference/), [docs/source/async.mdx](../docs/source/async.mdx)
  - 官方 RTC 真机示例：[examples/rtc/eval_with_real_robot.py](../examples/rtc/eval_with_real_robot.py)
- 未确认：就当前仓库而言，哪一条才应该被视为“主部署入口”。这需要结合你实际硬件环境、延迟约束和稳定性再判断。
- 已确认：机器人实例最终都通过 [src/lerobot/robots/utils.py](../src/lerobot/robots/utils.py) 的 `make_robot_from_config` 创建。

### 3.5 数据转换入口

#### 脚本族概览

| 脚本 | 主要场景 | 优先级 |
| --- | --- | --- |
| [convert_to_lerobot.py](../convert_to_lerobot.py) | 最基础的 joint+gripper 数据转换 | 中 |
| [convert_to_lerobot_change_shape.py](../convert_to_lerobot_change_shape.py) | 基础版 + 图像 resize | 中 |
| [convert_to_lerobot_sensorctrl.py](../convert_to_lerobot_sensorctrl.py) | 用 sensor gripper 作为控制量 | 高 |
| [convert_to_lerobot_sensorctrl_resized.py](../convert_to_lerobot_sensorctrl_resized.py) | sensorctrl + resize | 中 |
| [convert_to_lerobot_sensorctrl_gripperobs_resized.py](../convert_to_lerobot_sensorctrl_gripperobs_resized.py) | gripper observation 与 gripper action 分离 | 高 |
| [convert_to_lerobot_sensorctrl_gripperobs_resized_fps10.py](../convert_to_lerobot_sensorctrl_gripperobs_resized_fps10.py) | 上一版 + 下采样到 10Hz，和当前自定义推理脚本最吻合 | 高 |
| [convert_to_lerobot_resized_tcpquat.py](../convert_to_lerobot_resized_tcpquat.py) | 末端位姿四元数表示 | 高 |
| [convert_to_lerobot_resized_tcp6d.py](../convert_to_lerobot_resized_tcp6d.py) | 末端位姿 6D rotation 表示 | 高 |
| [convert_to_lerobot_umi.py](../convert_to_lerobot_umi.py) | UMI 数据转换，含路径修复与 resize | 高 |
| [convert_to_lerobot_umi_big.py](../convert_to_lerobot_umi_big.py) | UMI 不缩放版 | 中 |
| [convert_to_lerobot_umi copy.py](../convert_to_lerobot_umi copy.py) | 旧副本/实验副本 | 低 |

- 已确认：这批脚本至少已经被用于产出可供 ACT 与 Diffusion 训练配置消费的数据集，因为仓库里同时存在对应的训练配置文件、输出目录命名和推理脚本路径。
- 未确认：这些脚本是否已经形成“效果稳定、泛化足够好”的数据生产链；从仓库结构只能看出它们被用过，不能从代码本身推出真机效果良好。

#### 如何判断优先关注哪个

- 高概率推测：如果后续工作围绕你当前这套 Diana + diffusion + 10Hz checkpoint 的本地链路，应优先看 [convert_to_lerobot_sensorctrl_gripperobs_resized_fps10.py](../convert_to_lerobot_sensorctrl_gripperobs_resized_fps10.py)、[configs/train_config.json](../configs/train_config.json)、[run_inference.sh](../run_inference.sh)。
- 已确认：如果后续工作围绕 TCP 位姿控制，应优先看 [convert_to_lerobot_resized_tcpquat.py](../convert_to_lerobot_resized_tcpquat.py) / [convert_to_lerobot_resized_tcp6d.py](../convert_to_lerobot_resized_tcp6d.py) 与相应训练配置。
- 高概率推测：`convert_to_lerobot_sensorctrl_gripperobs_resized.py` 是 `fps10` 版本的前身，当前实际在线路更偏向 `fps10` 版本。

### 3.6 数据检查 / 调试入口

- [check_data_accuracy.py](../check_data_accuracy.py)：确认 parquet 是否含 `observation.state`。
- [look_parquet.py](../look_parquet.py)：手工查看 parquet 字段与某帧 state。
- [test_video.py](../test_video.py)：遍历 mp4 解码是否损坏。
- [src/lerobot/scripts/lerobot_dataset_viz.py](../src/lerobot/scripts/lerobot_dataset_viz.py)：正式的数据集可视化工具。

## 4. 核心模块说明

### 4.1 `src/lerobot/` 下的重要 package

| 模块 | 作用 | 备注 |
| --- | --- | --- |
| `configs/` | 训练、评测、策略、解析配置 | 通用主干 |
| `datasets/` | LeRobotDataset 格式、metadata、视频、统计、编辑工具 | 数据核心 |
| `policies/` | ACT / Diffusion / VQBeT / Pi0 / SmolVLA / RTC 等 | 模型核心 |
| `processor/` | 预处理、后处理、归一化、重命名、pipeline | 数据流桥梁 |
| `robots/` | 机器人抽象与硬件适配层 | 硬件核心 |
| `cameras/` | OpenCV / RealSense / ZMQ / Reachy2 camera 抽象 | 感知接入 |
| `async_inference/` | gRPC 异步远程推理 | 当前项目关键 |
| `scripts/` | 官方 CLI 入口实现 | 外层入口 |
| `teleoperators/` | leader device / phone / keyboard / gamepad | 采集与控制 |
| `envs/` | 仿真环境封装 | 评测/训练配套 |
| `transport/` | gRPC proto 与传输工具 | async 基础设施 |
| `utils/` | 通用工具、日志、训练辅助、控制辅助 | 基础设施 |

### 4.2 与 policy / dataset / camera / robot / inference / 训练流程有关的关系

```text
TrainConfig / EvalConfig
  -> datasets.factory.make_dataset
  -> policies.factory.make_policy
  -> policies.factory.make_pre_post_processors
  -> processor pipeline
  -> policy.forward / policy.select_action

Robot.get_observation()
  -> async_inference.helpers/raw_observation_to_observation
  -> preprocessor
  -> policy
  -> postprocessor
  -> robot.send_action()
```

### 4.3 与 dataset 有关的核心模块

- [src/lerobot/datasets/lerobot_dataset.py](../src/lerobot/datasets/lerobot_dataset.py)
  - 已确认：定义 `LeRobotDatasetMetadata` 与 `LeRobotDataset`。
  - 已确认：支持从本地目录或 HF Hub 读取 `meta/info.json`、`tasks.parquet`、`episodes/*.parquet`、`data/*.parquet`、`videos/*.mp4`。
- [src/lerobot/datasets/factory.py](../src/lerobot/datasets/factory.py)
  - 已确认：训练阶段统一构造 dataset，并解析 delta timestamps。
- [src/lerobot/datasets/utils.py](../src/lerobot/datasets/utils.py)
  - 已确认：定义默认路径模式、feature 元数据、parquet/hf dataset 辅助函数。
- [src/lerobot/datasets/dataset_tools.py](../src/lerobot/datasets/dataset_tools.py)
  - 已确认：数据集编辑工具，配合 `lerobot-edit-dataset` 使用。

### 4.4 与 policy 有关的核心模块

- [src/lerobot/policies/factory.py](../src/lerobot/policies/factory.py)
  - 已确认：策略类型到模型类/processor 的总工厂。
- [src/lerobot/policies/diffusion/](../src/lerobot/policies/diffusion/)
  - 已确认：当前项目本地训练与推理最相关。
- [src/lerobot/policies/utils.py](../src/lerobot/policies/utils.py)
  - 已确认：包含 `build_inference_frame`、`make_robot_action`，被本地推理脚本直接使用。
- 其他策略：
  - ACT、VQBeT、Pi0、Pi0.5、SmolVLA、Groot、XVLA、WallX、SAC、TDMPC
  - 已确认：都保留在仓库中，但不是当前 Diana 线路的首要入口。

### 4.5 与 processor / 训练流水线有关的核心模块

- [src/lerobot/processor/pipeline.py](../src/lerobot/processor/pipeline.py)
  - 已确认：定义可序列化处理流水线与 step registry。
- [src/lerobot/processor/observation_processor.py](../src/lerobot/processor/observation_processor.py)
  - 已确认：将原始 observation 标准化成 LeRobot 约定键名。
- [src/lerobot/processor/policy_robot_bridge.py](../src/lerobot/processor/policy_robot_bridge.py)
  - 已确认：提供 policy tensor <-> robot action dict 的桥接 step。
- [src/lerobot/scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py)
  - 已确认：训练主循环入口。

### 4.6 与机器人 / 相机有关的核心模块

- [src/lerobot/robots/robot.py](../src/lerobot/robots/robot.py)：统一 robot 抽象。
- [src/lerobot/robots/utils.py](../src/lerobot/robots/utils.py)：robot factory。
- [src/lerobot/cameras/](../src/lerobot/cameras/)：OpenCV / RealSense / ZMQ 等。
- [src/lerobot/scripts/lerobot_record.py](../src/lerobot/scripts/lerobot_record.py)：采集时组合 robot / teleop / dataset。

### 4.7 Diana 机械臂、相机、传感器适配层

- 已确认：Diana 适配主实现是 [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py)。
- 已确认：配置类是 [src/lerobot/robots/diana_follower/config_diana_follower.py](../src/lerobot/robots/diana_follower/config_diana_follower.py)。
- 已确认：该实现同时桥接两套系统：
  - Diana SDK：关节/笛卡尔控制
  - ROS2：图像订阅、夹爪状态订阅、夹爪控制发布
- 已确认：支持 `joint`、`pose`、`pose_quat`、`pose_6d` 四种控制模式。
- 已确认：仓库中还有多个并存变体：
  - `diana_follower1.py`
  - `diana_follower_ok.py`
  - `diana_follower_3_5.py`
  - `diana_followerjoint.py`
- 高概率推测：这些是演化过程中保留下来的实验版本，当前真正被 import 的是 `__init__.py` 指向的 `diana_follower.py`。

## 5. 与当前项目场景最相关的链路

### 5.1 数据集生成 / 转换链路

#### 当前最相关主链

```text
本地原始 episode 目录
  -> data.hdf5 + 图像路径
  -> convert_to_lerobot_*.py
  -> 生成 LeRobotDataset v3.0:
     meta/info.json
     meta/stats.json
     meta/tasks.parquet
     meta/episodes/chunk-xxx/file-xxx.parquet
     data/chunk-xxx/file-xxx.parquet
     videos/<camera>/chunk-xxx/file-xxx.mp4
```

- 已确认：转换脚本自己生成 `info.json` / `stats.json` / episode parquet / data parquet / videos，未调用上游 dataset builder。
- 已确认：所有转换脚本都把 `observation.state` 与 `action` 写入 parquet，同时把图像编码成 mp4。
- 已确认：脚本通常将 `action = state.copy()`，或只在 gripper 上区分 observation/action。
- 高概率推测：这些脚本服务于一个特定的数据采集目录格式，而不是通用转换工具，因此迁移性有限。

#### 关键变体差异

- `sensorctrl` 系列：
  - 已确认：使用 `pikaSensor` 作为 gripper action/control。
- `gripperobs` 系列：
  - 已确认：将 gripper state 与 gripper action 分开，`state` 来自 `pikaGripper`，`action` 来自 `pikaSensor`。
- `fps10` 版本：
  - 已确认：会对 state/action/图像做下采样，适配 10Hz 在线控制。
- `tcpquat` / `tcp6d` 版本：
  - 已确认：将原始 end-effector pose 转成四元数或 6D rotation 表示。
- `umi` 系列：
  - 已确认：读取 `localization/pose` 或兼容 `arm/endPose`，机器人类型标记为 `umi_agilex`。

### 5.2 模型训练链路

```text
configs/train_config*.json
  -> lerobot-train / lerobot_train.py
  -> TrainPipelineConfig.validate()
  -> datasets.factory.make_dataset()
  -> policies.factory.make_policy()
  -> policies.factory.make_pre_post_processors()
  -> DataLoader
  -> update_policy()
  -> outputs/train/<job>/checkpoints/<step>/pretrained_model
```

- 已确认：本地配置主要在训练 diffusion，少量 ACT。
- 已确认：训练数据多为本地 `data/lerobot_dataset_*`。
- 已确认：输出路径约定为 `outputs/train/.../checkpoints/.../pretrained_model`，这和 [run_inference.sh](../run_inference.sh) / [diffusion_using_diana.py](../diffusion_using_diana.py) 的模型路径吻合。

### 5.3 模型推理链路

#### 同步推理

```text
Robot.get_observation()
  -> clean_robot_observation()
  -> build_inference_frame()
  -> preprocessor
  -> DiffusionPolicy.select_action()
  -> postprocessor
  -> make_robot_action()
  -> adapt_action_for_diana()
  -> Robot.send_action()
```

证据：[diffusion_using_diana.py](../diffusion_using_diana.py)

#### 异步推理

```text
robot_client:
  Robot.get_observation()
  -> pickle + gRPC SendObservations

policy_server:
  raw_observation_to_observation()
  -> preprocessor
  -> policy.predict_action_chunk()
  -> postprocessor
  -> TimedAction chunk
  -> gRPC GetActions

robot_client:
  action queue aggregate
  -> robot.send_action()
```

证据：[src/lerobot/async_inference/robot_client.py](../src/lerobot/async_inference/robot_client.py), [src/lerobot/async_inference/policy_server.py](../src/lerobot/async_inference/policy_server.py), [src/lerobot/transport/services.proto](../src/lerobot/transport/services.proto)

### 5.4 机器人执行链路

- 已确认：在线执行最终都落到 `Robot.send_action()`。
- 已确认：Diana 执行时：
  - `joint` 模式用 `servoJ_ex`
  - `pose*` 模式用 `servoL_ex`
  - gripper 经 ROS2 `JointState` topic 发布
- 已确认：观测由 `get_observation()` 同时汇聚：
  - 机械臂关节或 TCP pose
  - gripper state
  - `cam_high` / `cam_global` 图像

### 5.5 观测构造 -> 模型输出 -> action 后处理 -> 下发机器人

#### 当前最相关的 Diana + diffusion 路径

1. `DianaFollower.get_observation()` 产出 flat dict。
2. 观测键大致包括：
   - `joint_i.pos` 或 `ee_*`
   - `gripper` 或 `gripper.pos`
   - `cam_high`
   - `cam_global`
3. 推理前会被转换成 LeRobot 规范键：
   - `observation.state`
   - `observation.images.cam_high`
   - `observation.images.cam_global`
4. preprocessor 做归一化、设备迁移、必要的图像堆叠。
5. policy 输出 chunk 或单步 action tensor。
6. postprocessor 反归一化。
7. `make_robot_action()` 或客户端映射回 robot action dict。
8. Diana 适配层再把 pose / quat / 6d 映射成 SDK 调用格式。

证据：[src/lerobot/async_inference/helpers.py](../src/lerobot/async_inference/helpers.py), [src/lerobot/policies/utils.py](../src/lerobot/policies/utils.py), [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py)

### 5.6 相机 / gripper / 机械臂观测如何组织进样本或推理输入

- 已确认：数据集样本里视觉是 `observation.images.<camera_name>`。
- 已确认：低维状态通常折叠到一个向量 `observation.state`。
- 已确认：在本地转换脚本里，`observation.state` 的 names 由脚本手写：
  - joint 模式一般是 `joint_0...joint_n + gripper`
  - TCP quat / 6d 模式则是 `ee_x/ee_y/...`
- 已确认：推理时要求机器人相机名字与训练特征名字一致，否则会出现 feature mismatch。

## 6. 配置与依赖

### 6.1 安装方式

- 已确认：标准安装来自 [pyproject.toml](../pyproject.toml)，可 `pip install -e .` 或 `pip install lerobot`。
- 已确认：依赖 extras 按硬件和模型拆分，例如 `intelrealsense`、`reachy2`、`unitree_g1`、`smolvla`、`groot`、`async` 等。
- 已确认：Diana SDK 不是 `pyproject.toml` 的标准 extra，而是仓库内单独目录 [diana_sdk/](../diana_sdk/)。

### 6.2 关键依赖

- 通用：
  - PyTorch / torchvision / accelerate
  - datasets / huggingface-hub / diffusers
  - opencv-python-headless / av / imageio
  - draccus / wandb / rerun-sdk
- 机器人相关：
  - pyserial / hidapi / pyrealsense2 / reachy2_sdk / python-can
- 当前项目场景额外依赖：
  - ROS2 Humble，见 [run_inference.sh](../run_inference.sh)
  - `rclpy`, `sensor_msgs`, `cv_bridge`，见 [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py)
  - Diana SDK，见 [diana_sdk/README.md](../diana_sdk/README.md)
  - pytorch3d，用于 6D rotation 转换，见 `tcp6d` 转换脚本与 Diana pose_6d 实现

### 6.3 配置文件主要在哪里

- 上游/通用：
  - [pyproject.toml](../pyproject.toml)
  - [src/lerobot/configs/](../src/lerobot/configs/)
- 当前项目本地训练：
  - [configs/train_config.json](../configs/train_config.json)
  - [configs/train_config_act.json](../configs/train_config_act.json)
  - [configs/train_config_tcp.json](../configs/train_config_tcp.json)
  - [configs/train_config_tcp6d.json](../configs/train_config_tcp6d.json)
  - [configs/train_config_umi.json](../configs/train_config_umi.json)
- 当前项目推理部署：
  - [run_inference.sh](../run_inference.sh)
  - [diffusion_using_diana.py](../diffusion_using_diana.py)

### 6.4 常改参数

- 数据路径：`dataset.root`, `repo_id`, 原始 HDF5 `SOURCE_ROOT`, 输出 `OUTPUT_ROOT`
- 机器人硬件：`robot.type`, `robot.port`, ROS topic, camera JSON
- 模型：`policy.type`, `pretrained_name_or_path`, `POLICY_DEVICE`
- 特征定义：`input_features`, `output_features`, state 维度，camera 名称
- 时序：`fps`, `n_obs_steps`, `n_action_steps`, `horizon`, `ACTIONS_PER_CHUNK`
- 训练：`batch_size`, `steps`, `num_workers`, `optimizer`, `scheduler`

### 6.5 平台与硬件假设

- 已确认：Python >= 3.10，见 [pyproject.toml](../pyproject.toml)。
- 已确认：训练强烈假设 GPU 可用，当前本地配置几乎都写死 `device=cuda`。
- 已确认：Diana 部署链路强依赖 Ubuntu/Linux + ROS2 + 本地网络可访问机器人。
- 高概率推测：当前项目主要运行在 Ubuntu + NVIDIA GPU + ROS2 Humble + 局域网机械臂环境。
- 已确认：`train_lerobot.slurm` 也体现了 GPU 集群训练假设。

## 7. 数据格式与文件流

### 7.1 输入数据通常来自哪里

- 已确认：当前本地脚本主要输入是 episode 目录下的 `data.hdf5` 和图像路径字符串。
- 已确认：UMI 与 AgileX 数据的 HDF5 结构不完全一样，所以脚本做了多版本兼容。
- 已确认：在线推理输入来自真实机器人观测，而不是 dataset。

### 7.2 中间产物与最终产物

| 阶段 | 产物 |
| --- | --- |
| 原始采集 | `data.hdf5` + 图像文件 |
| 转换后 | LeRobotDataset 目录 |
| 训练中 | `outputs/train/<job>/checkpoints/<step>/` |
| 推理中 | gRPC observation/action chunk，日志，可能的 `picture/` 调试图 |

### 7.3 LeRobotDataset 目录结构

- 已确认：典型结构包含：
  - `meta/info.json`
  - `meta/stats.json`
  - `meta/tasks.parquet`
  - `meta/episodes/chunk-xxx/file-xxx.parquet`
  - `data/chunk-xxx/file-xxx.parquet`
  - `videos/<video_key>/chunk-xxx/file-xxx.mp4`
- 证据：
  - [src/lerobot/datasets/utils.py](../src/lerobot/datasets/utils.py)
  - 各 `convert_to_lerobot*.py`

### 7.4 parquet / 视频 / observation / action 结构

- 已确认：parquet 里常见列包括：
  - `observation.state`
  - `action`
  - `timestamp`
  - `episode_index`
  - `frame_index`
  - `task_index`
- 已确认：视觉通常不直接嵌入 data parquet，而是通过 `video_path` 元信息指向 MP4。
- 已确认：转换脚本会在 `info.json` 中记录 feature 的 `dtype`, `shape`, `names`。

### 7.5 checkpoint 产物

- 已确认：训练输出的 `pretrained_model` 用于推理加载。
- 已确认：预处理器/后处理器也会一起保存，见训练与示例代码。

## 8. 调试与测试

### 8.1 正式测试

- 已确认：`tests/` 覆盖广泛，尤其是：
  - dataset：`tests/datasets/`
  - processor：`tests/processor/`
  - async inference：`tests/async_inference/`
  - cameras：`tests/cameras/`
  - training/utils：`tests/training/`, `tests/utils/`
- 已确认：这些测试主要验证上游主干能力，不覆盖当前 Diana 本地改造。

### 8.2 更像临时调试脚本的内容

- 根目录：
  - [test_camera.py](../test_camera.py)
  - [test_dataset.py](../test_dataset.py)
  - [test_video.py](../test_video.py)
  - [test_quat.py](../test_quat.py)
  - [test_6drepresentation.py](../test_6drepresentation.py)
  - [look_parquet.py](../look_parquet.py)
  - [check_data_accuracy.py](../check_data_accuracy.py)
- 已确认：这些更像当前项目自写的手工验证脚本，不属于规范测试套，也不应与 `tests/` 下的正式测试混为一谈。

### 8.3 排障时优先看哪些文件

| 问题类型 | 先看 |
| --- | --- |
| 数据列不对 / shape 不对 | `convert_to_lerobot*.py`, [check_data_accuracy.py](../check_data_accuracy.py), [look_parquet.py](../look_parquet.py) |
| 训练 feature mismatch | [configs/train_config*.json](../configs/), [src/lerobot/policies/factory.py](../src/lerobot/policies/factory.py), [src/lerobot/policies/utils.py](../src/lerobot/policies/utils.py) |
| 在线推理不动 / 动作延迟 | [run_inference.sh](../run_inference.sh), [src/lerobot/async_inference/robot_client.py](../src/lerobot/async_inference/robot_client.py), [src/lerobot/async_inference/policy_server.py](../src/lerobot/async_inference/policy_server.py) |
| 机器人观测异常 | [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py) |
| ROS2 / topic 问题 | [run_inference.sh](../run_inference.sh), [src/lerobot/robots/diana_follower/config_diana_follower.py](../src/lerobot/robots/diana_follower/config_diana_follower.py), `pika_ros` 环境 |
| 数据集可视化 | [src/lerobot/scripts/lerobot_dataset_viz.py](../src/lerobot/scripts/lerobot_dataset_viz.py) |

## 9. 风险点与技术债

### 9.1 重复脚本和命名混乱

- 已确认：根目录存在大量 `convert_to_lerobot*.py`，且命名约定只反映局部差异，没有统一入口。
- 已确认：Diana 目录下存在多个 `diana_follower*.py` 变体，容易误判哪个是真正生效版本。
- 已确认：根目录既有正式 CLI，也有一批直接运行脚本，入口分散。

### 9.2 实验性代码残留

- 已确认：`robot_client.py` 与 `policy_server.py` 中存在大量 `print(1111/2222/...)` 这类调试输出。
- 已确认：`policy_server.py` 中有对 diffusion 队列行为的“新增修复代码”直接写在主链路里，工程化程度一般。
- 已确认：`diana_follower.py` 中存在 `_save_debug_images()` 和 `picture/` 调试逻辑。

### 9.3 易误用入口

- 已确认：`README.md` 说的是上游通用用法，而当前仓库又混入了一批本地脚本；新接手的人容易把“你的自写链路”误认为“仓库唯一推荐链路”。
- 已确认：很多 `convert_to_lerobot*.py` 会直接 `shutil.rmtree(OUTPUT_ROOT)`，误改路径会覆盖已有数据。
- 已确认：训练配置里的 `repo_id` 有些写的是本地目录名而非 HF repo 规范，容易让新接手的人误以为都可以直接从 Hub 拉。
- 已确认：`configs/*`、`convert_to_lerobot*.py`、根目录 `test_*.py` 都更应视为当前项目自定义资产，而不是上游稳定接口。

### 9.4 环境耦合重

- 已确认：Diana 线路同时依赖 Python 环境、ROS2、`pika_ros`、Diana SDK、局域网 IP、相机设备索引。
- 已确认：异步推理需要 server/client 双端参数匹配，尤其 camera 名称、state 维度、模型特征。
- 高概率推测：一旦训练数据特征名和在线机器人观测名不一致，推理会直接失配。

### 9.5 测试缺口

- 已确认：没有看到 Diana 适配层的正式测试。
- 已确认：根目录转换脚本没有测试。
- 已确认：很多关键本地链路靠手工验证脚本，而不是自动化回归。

### 9.6 fork / upstream 风险

- 已确认：仓库存在 `origin` 与 `upstream` 双远程：
  - `origin = https://github.com/caizzzzy/lerobot.git`
  - `upstream = https://github.com/huggingface/lerobot.git`
- 已确认：这说明当前项目是从上游 fork 下来的，并保留了后续跟进上游更新的技术路径。
- 未确认：是否值得、以及何时跟进上游更新；这取决于你后续是否更看重新特性/修复，还是更看重本地改造的稳定性。
- 高概率推测：后续如果要跟进上游，冲突最重的区域会是：
  - `src/lerobot/async_inference/`
  - `src/lerobot/robots/diana_follower/`
  - 根目录自定义脚本与配置

## 10. 后续开发建议阅读路径

### 10.1 新 agent 最小阅读路径

1. 先看 [docs/PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)。
2. 再看 [pyproject.toml](../pyproject.toml) 和 [README.md](../README.md)，建立上游能力边界。
3. 看 [run_inference.sh](../run_inference.sh) 和 [diffusion_using_diana.py](../diffusion_using_diana.py)，理解当前项目自定义的真实运行方式。
4. 同时看 [docs/source/async.mdx](../docs/source/async.mdx) 和 [examples/rtc/eval_with_real_robot.py](../examples/rtc/eval_with_real_robot.py)，避免把“你的自写链路”误当成“仓库唯一推荐部署链路”。
5. 看 [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py)，理解硬件观测/动作定义。
6. 看 [configs/train_config.json](../configs/train_config.json) 与当前实际使用的数据转换脚本，理解训练特征。
7. 再回到：
   - [src/lerobot/async_inference/](../src/lerobot/async_inference/)
   - [src/lerobot/policies/diffusion/](../src/lerobot/policies/diffusion/)
   - [src/lerobot/datasets/](../src/lerobot/datasets/)

### 10.2 做训练时建议阅读

1. [configs/train_config*.json](../configs/)
2. [src/lerobot/scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py)
3. [src/lerobot/datasets/factory.py](../src/lerobot/datasets/factory.py)
4. [src/lerobot/policies/factory.py](../src/lerobot/policies/factory.py)
5. 对应 policy 子目录，如 [src/lerobot/policies/diffusion/](../src/lerobot/policies/diffusion/)

### 10.3 做推理/部署时建议阅读

1. [run_inference.sh](../run_inference.sh)
2. [diffusion_using_diana.py](../diffusion_using_diana.py)
3. [docs/source/async.mdx](../docs/source/async.mdx)
4. [src/lerobot/async_inference/robot_client.py](../src/lerobot/async_inference/robot_client.py)
5. [src/lerobot/async_inference/policy_server.py](../src/lerobot/async_inference/policy_server.py)
6. [examples/rtc/eval_with_real_robot.py](../examples/rtc/eval_with_real_robot.py)
7. [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py)

### 10.4 做数据转换时建议阅读

1. 当前目标脚本对应的 `convert_to_lerobot*.py`
2. [src/lerobot/datasets/utils.py](../src/lerobot/datasets/utils.py)
3. [src/lerobot/datasets/lerobot_dataset.py](../src/lerobot/datasets/lerobot_dataset.py)
4. [check_data_accuracy.py](../check_data_accuracy.py) / [look_parquet.py](../look_parquet.py) / [test_video.py](../test_video.py)

### 10.5 做机器人适配或硬件排障时建议阅读

1. [src/lerobot/robots/robot.py](../src/lerobot/robots/robot.py)
2. [src/lerobot/robots/utils.py](../src/lerobot/robots/utils.py)
3. [src/lerobot/robots/diana_follower/config_diana_follower.py](../src/lerobot/robots/diana_follower/config_diana_follower.py)
4. [src/lerobot/robots/diana_follower/diana_follower.py](../src/lerobot/robots/diana_follower/diana_follower.py)
5. [diana_sdk/README.md](../diana_sdk/README.md)

## 11. 已确认 / 高概率推测 / 未确认 汇总

### 11.1 已确认

- 当前仓库是上游 `lerobot` + 本地 Diana / 数据转换定制。
- 仓库中确实存在一条围绕你自定义脚本的本地链路：
  - `convert_to_lerobot*.py`
  - `configs/train_config*.json`
  - `run_inference.sh`
  - `diffusion_using_diana.py`
- Diana 适配层依赖 ROS2 + Diana SDK。
- 数据集格式为 LeRobotDataset v3.0，Parquet + MP4。
- 仓库同时保留了 `origin` 与 `upstream`，后续跟进上游在技术上是可行的。

### 11.2 高概率推测

- 当前在线实验更偏向使用 `sensorctrl_gripperobs_resized_fps10` 这一条数据链。
- `diana_follower.py` 是最终选定版本，其余 Diana 文件多为历史演化残留。
- 项目主要运行环境是 Ubuntu + CUDA GPU + ROS2 Humble。

### 11.3 未确认

- 当前团队实际日常使用的是同步推理脚本还是异步 server/client 链路，仓库里两条都存在。
- `run_inference.sh` 是否应该被视为“当前仓库最贴近真实部署的入口”。
- 真实原始 HDF5 数据采集流程是否也在本仓库中完成，还是来自外部系统后再导入。
- `pika_ros` 的具体职责与消息定义，因为它不在当前仓库内。
- 后续是否要跟进 `upstream`，以及合并成本是否值得。

## 12. 后续让 Codex 干活时建议引用本文件的方式

可以直接把下面这些提示语复用给 Codex：

- “先阅读 `docs/PROJECT_OVERVIEW.md`，只沿着当前 Diana + diffusion + fps10 数据链路排查，不要重新全仓扫描。”
- “以 `docs/PROJECT_OVERVIEW.md` 为上下文，帮我定位训练配置、数据转换脚本和在线推理入口之间是否一致。”
- “先按 `PROJECT_OVERVIEW` 里的‘推理/部署阅读路径’阅读，再修改 `run_inference.sh` 和 Diana 相关代码。”
- “参考 `PROJECT_OVERVIEW` 里对 `convert_to_lerobot*.py` 的分类，只检查最相关的脚本，不要把所有同名脚本都当成主入口。”
- “以 `PROJECT_OVERVIEW` 中记录的 feature 命名和数据流为准，帮我排查 observation/action shape mismatch。”
- “先根据 `PROJECT_OVERVIEW` 判断这是上游通用问题还是本地 Diana 定制问题，再决定改哪里。”
