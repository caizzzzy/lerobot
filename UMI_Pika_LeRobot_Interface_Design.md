# 《UMI-Pika-LeRobot 融合系统接口设计方案》

说明：当前工作区的 LeRobot 是 `0.4.4` 体系，主配置机制是 `draccus + dataclass + JSON`，不是 Hydra。所以下文提到的“配置适配”，应落到 LeRobot 的 `train/eval` 配置与数据特征声明上；如需 Hydra，只建议做一层生成器，不建议把主线重新做成 Hydra-first。

## 一、总体架构

```text
Pika/机械臂/相机 ROS 2 原始话题
-> RawCapture 层：MCAP + sidecar 标定/时间戳
-> AlignedEpisode 层：按统一 policy_fps 重采样，保留绝对位姿
-> UmiRelativeView 层：在 sample-time 生成 UMI 风格相对 obs/action
-> LeRobot Diffusion 训练
-> Checkpoint
-> ROS 2 在线闭环：ObsAssembler -> PolicyWorker -> ActionScheduler -> RobotBridge
```

核心决策有四条：

- 原始数据层永远保留绝对量，不在录制阶段改写成相对量。
- 真正送入模型的相对观测与相对动作，在“时间窗口已经组装完成之后”再计算。
- 第一阶段严格对齐 UMI，主视觉只用末端宽角 RGB；深度先录制、不进基线策略。
- 在线部署不直接“推理一次就立刻发一次动作”，而是走“动作块预测 + 时间戳调度 + 高频插值控制”。

## 二、Observation Space 的裁剪与同步

Pika 的多模态不同频：定位 `100 Hz` 级、gripper/IMU `50-100 Hz` 级、RGB/Depth `30 Hz` 级。这里不建议依赖 `message_filters` 做一次性同步，因为它对异频流、推理延迟、回放复现都不够稳。更合理的是统一采用“时间戳缓冲区 + 离线/在线同构重采样”：

- 每条 ROS 2 消息都保留两套时间：`source_stamp` 和 `host_rx_monotonic_stamp`。
- 录制层写入 MCAP，同时写入相机内参、外参、topic map、episode 边界、定位状态。
- 离线构建 `AlignedEpisode` 时，先选定统一 `policy_fps`，建议首版 `10 Hz`，稳定后再升到 `15-20 Hz`。
- 统一采样栅格 `t_k` 以“主视觉相机最近有效帧”为锚点。
- RGB/Depth 用最近邻取样；6-DoF 与 gripper 宽度用线性插值；定位失锁或超阈值缺测直接切段丢弃。
- 在线 `ObsAssembler` 必须复用与离线完全同一套插值规则，否则训练/部署分布会漂移。

观测映射建议如下：

| LeRobot key | 形状 | 来源 | 决策 |
| --- | --- | --- | --- |
| `observation.images.wrist_rgb` | `(3,224,224)` | Pika 末端宽角 RGB，经去畸变/裁剪/缩放 | 基线必选，直接对齐 UMI 视觉输入 |
| `observation.images.global_rgb` | `(3,224,224)` | 全局 RGB | 二阶段再加；首版可只录制不训练 |
| `observation.state` | `(16,)` | 由绝对 pose/gripper 在 sample-time 计算 | 作为 LeRobot Diffusion 的低维输入 |
| `raw.depth.*` | 原始分辨率 | Depth topic | 先保留到原始/对齐数据层，不进基线 UMI encoder |

`observation.state` 的单臂基线定义建议为：

- `eef_pos_rel`: 3 维，过去/当前 TCP 相对“当前时刻 TCP”的平移。
- `eef_rot6d_rel`: 6 维，过去/当前 TCP 相对“当前时刻 TCP”的旋转 6D 表示。
- `gripper_width`: 1 维，真实夹爪开口宽度，单位米。
- `eef_rot6d_wrt_start`: 6 维，相对 episode 起点姿态的旋转 6D。

合计 `16` 维。这里刻意不把 raw depth 直接塞进 `observation.state`，因为那会把 UMI 从“RGB + low-dim”变成另一套算法；如果后续要做多模态扩展，应单独加深度 encoder，而不是在第一阶段混入。

## 三、Action Space 转换机制

这里最关键的结论是：相对动作不要在 ROSbag 解析时就固化，也不要在最原始的 topic 录制层做。最合理的位置，是 `LeRobot 自定义 Dataset / Processor` 在已经拿到“当前观测窗口 + 未来动作块”之后，再把绝对动作块变成 UMI 风格相对动作块。

原因很直接：

- UMI 的相对动作不是“每一帧各自相对自己”，而是“整段未来 action chunk 都相对当前观测末帧”。
- 这个基准帧依赖 `n_obs_steps`、`delta_timestamps`、episode 边界和当前 sample。
- 如果在 ROSbag 解析阶段就转，相对基准会被过早固定，后续改 horizon、fps、obs window 都会失真。

因此建议分三层表示：

- `RawCapture`：保存 `/arm_end_pose` 或控制器 `ActualTCPPose` 的绝对 TCP 位姿，外加 gripper 宽度。
- `AlignedEpisode`：统一到 `policy_fps` 的绝对状态表。
- `UmiRelativeView`：在 `__getitem__` 或等效 processor 中，按当前 sample 计算相对 obs/action。

动作标签的“真值源”应优先选机械臂控制器返回的实际 TCP 位姿，例如 `/arm_end_pose` 或 SDK `ActualTCPPose`，而不是原始 `/pika_pose*`。`/pika_pose*` 更适合作为 teleop 上游和诊断量；只有在你已经有稳定的 `T_base_pika -> T_base_tcp` 标定，并确认控制器姿态不可用时，才应退化使用它。

LeRobot 里建议把模型输出定义为单臂 `10` 维：

| Slice | 维度 | 物理含义 |
| --- | --- | --- |
| `action[0:3]` | 3 | `T_current^-1 * T_target` 的平移分量，单位米 |
| `action[3:9]` | 6 | `T_current^-1 * T_target` 的旋转 6D 表示 |
| `action[9]` | 1 | 目标夹爪开口宽度，单位米，绝对量 |

补充两点：

- 夹爪建议用“真实开口宽度（m）”而不是编码器角度或 `[0,1]` 归一化值；硬件侧再做逆映射。
- 双臂时直接按 robot0、robot1 顺序拼成 `20` 维，不要改表示法。

## 四、控制频率、时间集成与闭环执行

部署侧建议拆成四个非阻塞组件，而不是一个大 ROS 2 节点串行做完：

- `SensorBuffer`：各 topic 回调只做入环形缓冲区，绝不等推理。
- `ObsAssembler`：按固定 `policy_fps` 取最近窗口，复用离线同构插值。
- `PolicyWorker`：异步 GPU 推理，只保留“最新观测任务”，旧任务直接丢弃。
- `ActionScheduler`：维护未来时间戳动作队列，以硬件伺服频率插值下发。

推荐的频率关系是：

- 采集层：RGB/Depth `30 Hz`，pose/gripper `100 Hz` 左右。
- 策略层：首版 `policy_fps = 10 Hz`。
- 模型：`n_obs_steps = 2`，`horizon = 16`，`n_action_steps = 4~6`。
- 硬件控制层：维持机械臂控制器原生伺服频率，通常 `50-125 Hz` 甚至更高。

时间集成的关键不是“每个控制 tick 都重新推理”，而是：

- 在时刻 `t_obs_last` 推出一个未来动作块 `a_0...a_{H-1}`。
- 给每个动作显式分配 `t_i = t_obs_last + i * dt_policy`。
- 仅保留 `t_i > now + exec_margin` 的未来动作，过期前缀直接丢掉。
- 下发给机器人时，再减去 `robot_action_latency` / `gripper_action_latency` 做补偿。
- `ActionScheduler` 用插值器在高频伺服层平滑跟踪这些 waypoint。

这比直接调用 LeRobot 的 `policy.select_action()` 更适合实机，因为后者有动作队列，但没有显式时间戳与延迟补偿语义。UMI 要复现的“时间集成”，本质上就是这套“预测块 + 未来调度 + 过期裁剪 + 高频插值”。

## 五、分步开发路线图

| 子任务 | 输入 | 输出 |
| --- | --- | --- |
| `1. 离线采集转换器` | MCAP/目录化采集数据、topic 配置、标定文件 | `AlignedEpisode`：统一 `policy_fps` 的绝对 pose/gripper/RGB/depth 索引与有效性掩码 |
| `2. UMI 相对视图 Dataset` | `AlignedEpisode`、`n_obs_steps`、`horizon`、相对表示配置 | 可供训练的 sample：`observation.images.*`、`observation.state`、`action` |
| `3. LeRobot 配置适配` | 观测/动作 schema、训练超参 | `draccus` JSON 配置、`input_features/output_features`、stats 生成逻辑 |
| `4. ROS 2 在线闭环推理节点` | 实时 topic、checkpoint、标定、延迟参数 | `ObsAssembler + PolicyWorker + ActionScheduler` 闭环链路 |
| `5. 实机验证与回放工具` | 推理节点输出、机器人桥接接口、日志目录 | 轨迹回放、延迟报告、丢帧统计、安全限幅与故障切换验证 |

验收顺序建议是：

1. 先验证“离线重采样后的视频/pose/gripper 对齐是否正确”。
2. 再验证“Dataset 取出的相对 obs/action 是否与 UMI 数学定义一致”。
3. 再跑 LeRobot 训练。
4. 最后接实机闭环，不要反过来。

## 六、关键确认问题

1. 你们当前可作为“动作真值源”的是哪个量：机械臂控制器返回的 `ActualTCPPose` / `/arm_end_pose`，还是只有 `/pika_pose*`？另外，`T_base_tcp` 与 `T_pika`/相机外参的标定结果是否已经稳定可用？
2. 首版训练你希望主视觉输入选哪一路：末端宽角 RGB、RealSense color，还是二者同时用？对应的 depth 是否与该主 RGB 已经完成硬件级或几何级对齐？
