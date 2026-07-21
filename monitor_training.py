#!/usr/bin/env python3
"""Monitor a background training job without interrupting it.

Example:
    python monitor_training.py --pid 2220 --checkpoint checkpoints/full/last.pt \
        --log logs/full_train.log --epochs 10
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import subprocess
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="Background training process ID.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/full/last.pt"))
    parser.add_argument("--log", type=Path, default=Path("logs/full_train.log"))
    parser.add_argument("--epochs", type=int, default=10, help="Total planned epochs, for progress display.")
    parser.add_argument("--interval", type=float, default=15, help="Refresh interval in seconds.")
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()
    except FileNotFoundError:
        return "unavailable"


def read_checkpoint(path: Path) -> str:
    if not path.exists():
        return "尚未完成第 1 轮；暂时没有检查点。"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = checkpoint.get("epoch", "?")
        history = checkpoint.get("history", [])
        if history:
            latest = history[-1]
            return (
                f"已完成第 {epoch} 轮；"
                f"train_loss={latest['train_loss']:.4f}，"
                f"val_loss={latest['val_loss']:.4f}"
            )
        return f"已保存第 {epoch} 轮检查点。"
    except Exception:
        return "检查点正在写入，下一次刷新会重新读取。"


def tail(path: Path, lines: int = 4) -> str:
    if not path.exists():
        return "（日志文件尚未创建）"
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:]) or "（日志暂时为空；Python 可能正在缓冲输出）"
    except OSError:
        return "（日志暂时无法读取）"


def render(args: argparse.Namespace) -> None:
    os.system("clear")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alive = Path(f"/proc/{args.pid}").exists()
    process = command_output(["ps", "-p", str(args.pid), "-o", "pid=,etime=,%cpu=,rss=,stat=,cmd="])
    gpu = command_output([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])

    print("古籍 LSTM · 实时训练监控")
    print("=" * 48)
    print(f"刷新时间：{now}（每 {args.interval:g} 秒刷新；Ctrl+C 只退出监控）")
    print(f"训练进程：{'运行中' if alive else '未找到/已结束'}（PID {args.pid}）")
    print(f"进程详情：{process or '无'}")
    print(f"GPU：{gpu or '无数据'}")
    print(f"训练进度：{read_checkpoint(args.checkpoint)} / 计划 {args.epochs} 轮")
    print("\n最近日志：")
    print(tail(args.log))


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    while True:
        render(args)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
