# 论文与配置对齐备忘

本文档用于快速回答本仓库中 ACT / Diffusion Policy 与原论文、LeRobot 当前实现、以及本项目真机场景之间的关系，避免后续重复从零核对。

## 范围

- 论文：
  - ACT: <https://arxiv.org/pdf/2304.13705>
  - Diffusion Policy: <https://arxiv.org/pdf/2303.04137v4>
- 本仓库实现：
  - [ACTConfig](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/act/configuration_act.py)
  - [DiffusionConfig](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py)
  - [DiffusionPolicy](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py)
  - [DatasetConfig](/mnt/nas/projects/robot/lerobot/src/lerobot/configs/default.py)
  - [dataset factory](/mnt/nas/projects/robot/lerobot/src/lerobot/datasets/factory.py)

## 状态标签

- `已确认`：能从论文原文或当前代码直接确认
- `高概率推测`：论文未直接写明，但结合实现和常规做法较合理
- `未确认`：当前没有足够依据，不应写成定论

## ACT 对齐结论

### 已确认

- ACT 论文表格给出的典型超参数包括：
  - `learning rate = 1e-5`
  - `batch size = 8`
  - `encoder layers = 4`
  - `decoder layers = 7`
  - `hidden dimension = 512`
  - `heads = 8`
  - `feedforward dimension = 3200`
  - `chunk size = 100`
  - `dropout = 0.1`
  - `beta (KL weight) = 10`
- LeRobot 当前 ACT 默认值基本按上面这套走，但 [configuration_act.py](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/act/configuration_act.py#L118) 明确说明：
  - 原 ACT 实现虽然配置为 `n_decoder_layers=7`
  - 但由于原实现 bug，实际只用了第一层
  - 因此 LeRobot 故意将其设为 `1` 以匹配原实现行为
- ACT 论文明确写到遥操作和数据记录发生在 `50Hz`

### 高概率推测

- 如果目标是“尽量接近原 ACT 实际跑出来的行为”，应优先保留 `n_decoder_layers=1`
- 如果目标是“照着论文表格字面写”，可以写 `7`，但这不一定更接近原作者真正跑出来的模型

### 未确认

- ACT 论文 PDF 中，我未确认到“必须使用 ImageNet 预训练 ResNet18”的直接原文
- 但 LeRobot 当前 ACT 默认配置确实使用了预训练 backbone，这属于“当前实现和复现实践已确认”，不是“论文原文已确认”

## Diffusion Policy 对齐结论

### 已确认

- Diffusion Policy 论文真机部分明确给出过这类设定：
  - `To = 2`
  - `Ta = 8`
  - `Tp = 16`
  - 图像输入 `320x240`
  - crop 到 `288x216`
  - `lr = 1e-4`
  - `weight decay = 1e-6`
  - 训练 diffusion 迭代数 `100`
  - 推理迭代数 `16`
- 论文正文明确写到：
  - policy 以 `10Hz` 输出机器人命令
  - 再线性插值到 `125Hz` 给低层执行
- 论文原文明确写到：
  - 多相机使用 `separate encoders`
  - 图像编码器使用 `ResNet18`
  - 不使用预训练
  - 用 `Spatial Softmax`
  - 用 `GroupNorm` 替换 `BatchNorm`

### 对 LeRobot 当前实现的影响

- [modeling_diffusion.py](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py#L203) 支持两种视觉编码方式：
  - `use_separate_rgb_encoder_per_camera=true`：每个相机一个 encoder
  - `use_separate_rgb_encoder_per_camera=false`：多相机共享一个 encoder
- 论文一致性上，`true` 更贴近原文
- 参数量控制和小数据集正则化上，`false` 可能更稳，但这是工程折中，不是论文默认

### 需要特别注意的点

- [configuration_diffusion.py](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py#L133) 中的 `down_dims=(512, 1024, 2048)` 是 LeRobot 当前默认值
- 这个值不能直接写成“论文真机表格已确认”
- 论文中可以直接确认的是观测步数、动作步数、预测 horizon、图像尺寸、学习率、迭代步数等；`down_dims` 更像实现细节

## DDPM / DDIM：当前仓库里到底怎么生效

### 已确认

- [modeling_diffusion.py](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py#L215) 只创建了一个 `noise_scheduler`
- 训练时：
  - [compute_loss](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py#L333) 会用这个 scheduler 的 `add_noise(...)`
- 推理时：
  - [conditional_sample](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py#L232) 会用同一个 scheduler 的 `set_timesteps(...)` 和 `step(...)`

这意味着当前实现里：

- `noise_scheduler_type="DDPM"`：训练和推理都按 DDPM scheduler 走
- `noise_scheduler_type="DDIM"`：训练和推理都按 DDIM scheduler 走

### 重要结论

- 当前**不能只靠 config**精确表达“训练 DDPM、推理 DDIM”
- 如果目标是“尽量贴近论文真机做法”，理想实现应当是：
  - 训练 scheduler：DDPM
  - 推理 scheduler：DDIM
  - 两者共享同一套 beta schedule / prediction type / train timesteps

### 工程建议

- 不改代码时，如果你只想让部署时走 DDIM：
  - 可以在推理加载时把保存下来的 policy config 改成 `noise_scheduler_type="DDIM"` 并设置 `num_inference_steps=16`
  - 这通常不需要改模型权重本身，只需要改加载配置
- 但这不是严格的“训练 DDPM、推理 DDIM”复现，而是“训练阶段保存的是 DDPM 配置，推理阶段用 DDIM scheduler 重建采样器”

### 高概率推测

- 对于当前 epsilon-prediction 的实现，权重本身通常不是“只属于 DDPM 不能用于 DDIM”的
- 真正敏感的是：
  - `beta_schedule`
  - `num_train_timesteps`
  - `prediction_type`
  - 推理时的 scheduler 类型和步数

### 实现等价性 vs 论文语义

这一点很容易混淆，需要单独说明。

- 从**当前仓库实现**看：
  - 训练阶段 [compute_loss](/mnt/nas/projects/robot/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py#L333) 只依赖 scheduler 的 `add_noise(...)`
  - 在 `diffusers` 中，`DDPMScheduler.add_noise(...)` 与 `DDIMScheduler.add_noise(...)` 使用的是同类 forward noising 公式
  - 因此当以下参数保持一致时：
    - `num_train_timesteps`
    - `beta_schedule`
    - `beta_start`
    - `beta_end`
    - `prediction_type`
  - 那么把 `noise_scheduler_type` 设为 `DDPM` 或 `DDIM`，训练阶段的差异通常很小，主要差别集中在推理 `step(...)`

- 从**论文设定语义**看：
  - Diffusion Policy 论文对外描述更接近“训练按 DDPM / iDDPM 设定，真机推理时用 DDIM 减少采样步数”
  - 所以如果问题是“哪种写法更贴论文原意”，更应写成：
    - 训练：`DDPM`
    - 推理：`DDIM + 16 steps`

换句话说：

- “当前代码里训练几乎等价”
- 不等于
- “论文语义上这两种配置写法完全等价”

### 对当前项目的直接结论

- 如果你只关心当前代码能否跑通、效果是否足够好：
  - 直接用 `DDIM` 训练和部署，通常不是大问题
- 如果你想让实验描述、配置归因、论文对齐更清晰：
  - 更推荐记录为“训练 DDPM，部署 DDIM”
- 如果你后续要做更严格的论文对齐实验，最理想的实现仍然是把训练 scheduler 和推理 scheduler 显式拆开

## 频率选择建议

### 已确认

- ACT 论文：数据记录和遥操作是 `50Hz`
- Diffusion Policy 真机：policy 输出是 `10Hz`

### 对当前项目的建议

- 如果只做一份数据并优先让 diffusion 真机跑顺：
  - 优先保证稳定同步的 `10Hz`
- 如果希望同时兼顾 ACT：
  - 更理想的方式是保留更高频原始数据，再导出低频版本

### 推荐方案

- 如果硬件做不到 `50Hz`，可以采用：
  - 原始采集 `30Hz`
  - 再离线导出 `10Hz` 版本用于 diffusion
- 这比“只采一个 10Hz 版本”更有上限，因为：
  - `30Hz -> 10Hz` 可以下采样
  - `10Hz -> 30Hz` 不能恢复高频细节

### 需要牢记的一点

- ACT 的 `chunk_size=100`
  - 在 `50Hz` 下对应 2 秒动作块
  - 在 `30Hz` 下对应约 3.3 秒动作块
  - 在 `10Hz` 下对应 10 秒动作块

所以一旦采样频率变了，ACT 里 `chunk_size=100` 的实际时间含义也变了。

## image_transforms 是什么，是否必要

### 已确认

- `image_transforms` 是 dataset 层的数据增强配置，不是某个 policy 独有功能
- [DatasetConfig](/mnt/nas/projects/robot/lerobot/src/lerobot/configs/default.py#L23) 默认就带有 `image_transforms`
- [dataset factory](/mnt/nas/projects/robot/lerobot/src/lerobot/datasets/factory.py#L83) 会在加载训练数据时把它传进 dataset
- [LeRobotDataset](/mnt/nas/projects/robot/lerobot/src/lerobot/datasets/lerobot_dataset.py#L1070) 会在取样时，把 transform 应用到所有相机图像
- [lerobot-dataset-v3.mdx](/mnt/nas/projects/robot/lerobot/docs/source/lerobot-dataset-v3.mdx#L154) 明确写了：
  - transforms 只在训练时应用
  - 不在录制时写进原始数据

### 和 ACT / Diffusion 的关系

- ACT 没有自己的 `crop_shape` 图像裁剪模块，所以很多视觉变化只能靠：
  - 原始数据多样性
  - dataset-level `image_transforms`
- Diffusion 本身在 policy 内部已经有：
  - `crop_shape`
  - `random crop`
  - `GroupNorm`
  - 多相机 encoder

所以你会感觉：

- ACT 文档里不怎么强调 `image_transforms`
- Diffusion 的配置里经常看得到和图像处理相关的项

这不表示：

- ACT 不能用 transforms
- Diffusion 必须用 transforms

### 实用建议

- 对真实机器人数据，`image_transforms` 不是必需项
- 但当你存在这些问题时，它通常值得试：
  - 光照变化
  - 白平衡波动
  - 相机轻微曝光变化
  - 背景差异

### 推荐强度

- 第一轮训练建议保守：
  - brightness: `0.9-1.1`
  - contrast: `0.9-1.1`
  - saturation: `0.9-1.1`
  - hue: `-0.02 ~ 0.02`
- 不建议一开始就上很强的 blur / rotation
- 操作型任务里几何增强过重，容易破坏真实相机-机械臂几何关系

## 真机 val / test 怎么切

### 已确认

- 真机数据应按 `episode` 切，而不是按 `frame` 切
- [lerobot_edit_dataset.py](/mnt/nas/projects/robot/lerobot/src/lerobot/scripts/lerobot_edit_dataset.py#L39) 已支持：
  - 按比例切分
  - 按 episode index 显式切分

### 正确理解

- 一整条轨迹只能属于 `train` 或 `val` 或 `test` 其中一个
- 不能把同一条轨迹前半段给 train、后半段给 val

## 目前最值得记住的简版结论

- ACT：
  - `n_decoder_layers=1` 在 LeRobot 中是有意为之，用来贴近原 ACT 实际实现
  - `50Hz` 是论文时间尺度
- Diffusion：
  - 论文真机是 `To=2, Ta=8, Tp=16`
  - 多相机 encoder 在论文里是分开的
  - 真机 policy 时间尺度是 `10Hz`
- 当前仓库：
  - `noise_scheduler_type` 同时影响训练和推理
  - 不改代码时，不能纯靠 config 实现“训练 DDPM、推理 DDIM”
  - `image_transforms` 是 dataset 层训练增强，不是某个 policy 专属开关
