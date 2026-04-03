#!/usr/bin/env bash

set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <pid> <log_path> [tmux_target] [output_dir] [interval_seconds]" >&2
    exit 1
fi

PID="$1"
LOG_PATH="$2"
TMUX_TARGET="${3:-}"
OUTPUT_DIR="${4:-logs/train_watch_$(date +%Y%m%d_%H%M%S)_pid${PID}}"
INTERVAL="${5:-30}"
START_TS="$(date '+%F %T %z')"
START_EPOCH="$(date +%s)"

mkdir -p "$OUTPUT_DIR"

SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
touch "$SUMMARY_FILE"

log_summary() {
    printf '[%s] %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$SUMMARY_FILE"
}

write_section() {
    local file="$1"
    local title="$2"
    {
        printf '\n===== %s @ %s =====\n' "$title" "$(date '+%F %T %z')"
        cat
    } >>"$file"
}

capture_cmd() {
    local file="$1"
    shift
    {
        "$@"
    } 2>&1 | write_section "$file" "$*"
}

capture_shell() {
    local file="$1"
    shift
    {
        bash -lc "$*"
    } 2>&1 | write_section "$file" "$*"
}

snapshot() {
    local tag="$1"
    local snap_dir="$OUTPUT_DIR/$tag"
    mkdir -p "$snap_dir"

    date '+%F %T %z' >"$snap_dir/timestamp.txt"

    capture_shell "$snap_dir/process.txt" "ps -p $PID -o pid,ppid,pgid,sid,etimes,%cpu,%mem,stat,lstart,cmd"
    capture_shell "$snap_dir/process.txt" "ps --ppid $PID -o pid,ppid,pgid,etimes,%cpu,%mem,stat,lstart,cmd"

    if [ -d "/proc/$PID" ]; then
        capture_shell "$snap_dir/proc.txt" "tr '\\0' ' ' </proc/$PID/cmdline"
        capture_shell "$snap_dir/proc.txt" "sed -n '1,120p' /proc/$PID/status"
        capture_shell "$snap_dir/proc.txt" "sed -n '1,120p' /proc/$PID/limits"
        capture_shell "$snap_dir/proc.txt" "grep -E 'Vm|Threads|SigQ|Cpus_allowed_list|Mems_allowed_list|State|voluntary_ctxt_switches|nonvoluntary_ctxt_switches' /proc/$PID/status"
    else
        log_summary "proc directory for pid $PID no longer exists during snapshot $tag"
    fi

    capture_shell "$snap_dir/system.txt" "free -h"
    capture_shell "$snap_dir/system.txt" "df -h ."

    if command -v nvidia-smi >/dev/null 2>&1; then
        capture_shell "$snap_dir/gpu.txt" "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader"
        capture_shell "$snap_dir/gpu.txt" "nvidia-smi pmon -c 1"
    fi

    if [ -f "$LOG_PATH" ]; then
        capture_shell "$snap_dir/log_tail.txt" "tail -n 200 '$LOG_PATH'"
        if rg -n -m 20 'Segmentation fault|Traceback|RuntimeError|ERROR|CUDA error|DataLoader worker|killed by signal|bus error|core dumped' "$LOG_PATH" >"$snap_dir/log_matches.txt" 2>/dev/null; then
            log_summary "log pattern matches found during snapshot $tag"
        fi
    fi

    if [ -n "$TMUX_TARGET" ] && command -v tmux >/dev/null 2>&1; then
        capture_shell "$snap_dir/tmux.txt" "tmux capture-pane -pt '$TMUX_TARGET' -S -120"
    fi

    if command -v journalctl >/dev/null 2>&1; then
        capture_shell "$snap_dir/journal.txt" "journalctl --since '@$START_EPOCH' --no-pager -n 200"
    fi
}

log_summary "monitor start pid=$PID log_path=$LOG_PATH tmux_target=${TMUX_TARGET:-<none>} interval=${INTERVAL}s output_dir=$OUTPUT_DIR start=$START_TS"
snapshot "start"

while kill -0 "$PID" 2>/dev/null; do
    sleep "$INTERVAL"
    snapshot "alive_$(date +%Y%m%d_%H%M%S)"
done

log_summary "pid $PID is no longer alive; collecting final snapshot"
snapshot "after_exit"
log_summary "monitor finished for pid $PID"
