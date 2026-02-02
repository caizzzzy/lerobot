#!/bin/bash

# ==========================================
# 用户配置区域 (请根据你的环境修改以下变量)
# ==========================================

# --- 网络配置 ---
SERVER_HOST="0.0.0.0"
CLIENT_HOST="192.168.10.134"
PORT="8080"

# --- 机器人配置 (Robot) ---
ROBOT_TYPE="diana_follower"              # 你的机器人类型
ROBOT_PORT="192.168.10.76"              # 你的机器人连接端口 
# ROBOT_ID="diana_robot"                # 机器人ID，用于读取校准文件

# # --- 摄像头配置 (Camera) ---
# # 注意：确保JSON格式正确，尤其是引号
# CAMERAS="{ laptop: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, phone: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}"
CAMERAS="{cam_high: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},cam_fish: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},cam_global: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
# CAMERAS="{cam_high: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
# --- 策略与模型配置 (Policy) ---
POLICY_TYPE="diffusion"                      # 策略类型 (例如: act, smolvla, diffusion 等)
MODEL_PATH="/mnt/nas/projects/robot/lerobot/outputs/train/picknput128/checkpoints/200000/pretrained_model"       # 服务器上的模型路径或 HuggingFace ID
# TASK="dummy"                             # 任务名称 (部分策略不需要)
POLICY_DEVICE="cuda"                      # 推理设备: 'cuda', 'mps' (Mac), or 'cpu'
ACTIONS_PER_CHUNK=50                     # 每次推理输出的动作数量

# --- 客户端微调参数 (Client Tuning) ---
CHUNK_SIZE_THRESHOLD=0.5                 # 发送新观测数据的阈值
AGGREGATE_FN="weighted_average"          # 动作聚合函数
DEBUG_VISUALIZE=True                     # 是否可视化队列大小 (调试用)

# ==========================================
# 脚本逻辑 (通常无需修改)
# ==========================================

MODE=$1

if [ "$MODE" == "server" ]; then
    echo "=========================================="
    echo "正在启动策略服务器 (Policy Server)..."
    echo "监听地址: $SERVER_HOST:$PORT"
    echo "=========================================="
    
    python -m lerobot.async_inference.policy_server \
        --host="$SERVER_HOST" \
        --port="$PORT"

elif [ "$MODE" == "client" ]; then
    echo "=========================================="
    echo "正在启动机器人客户端 (Robot Client)..."
    echo "连接至: $CLIENT_HOST:$PORT"
    echo "机器人: $ROBOT_TYPE ($ROBOT_ID)"
    echo "模型: $MODEL_PATH ($POLICY_TYPE)"
    echo "=========================================="

    python -m lerobot.async_inference.robot_client \
        --server_address="$CLIENT_HOST:$PORT" \
        --robot.type="$ROBOT_TYPE" \
        --robot.cameras="$CAMERAS" \
        --policy_type="$POLICY_TYPE" \
        --pretrained_name_or_path="$MODEL_PATH" \
        --policy_device="$POLICY_DEVICE" \
        --actions_per_chunk="$ACTIONS_PER_CHUNK" \
        --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD" \
        --aggregate_fn_name="$AGGREGATE_FN" \
        --debug_visualize_queue_size="$DEBUG_VISUALIZE" \
        --robot.port="$ROBOT_PORT" \
        # --robot.id="$ROBOT_ID" \
        
        # --task="$TASK" \

else
    echo "用法错误。"
    echo "启动服务器: ./run_async.sh server"
    echo "启动客户端: ./run_async.sh client"
fi