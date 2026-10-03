"""視線推定の負荷を測る。システム全体の使用率ではなく、このプロセス自身のCPU時間で測る

    uv run python tools/bench.py            # 60秒
    uv run python tools/bench.py --seconds 20
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from gaze_overlay.camera import Camera
from gaze_overlay.config import load_config
from gaze_overlay.tracker import GazeTracker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    args = ap.parse_args()

    cfg = load_config()
    tracker = GazeTracker(cfg.tracker, cfg.resolve(cfg.tracker.face_model))
    cam = Camera(cfg.camera)
    if not cam.open():
        raise SystemExit("カメラを開けませんでした")
    info = cam.info
    print(f"カメラ: {info.backend} {info.width}x{info.height} → 推論 {info.process_width}x{info.process_height}")
    for _ in range(10):
        cam.read()

    proc, frames, faces, blinks, downs = [], 0, 0, 0, 0
    t0 = time.perf_counter()
    c0 = time.process_time()
    while time.perf_counter() - t0 < args.seconds:
        frame = cam.read()
        if frame is None:
            continue
        obs = tracker.process(frame)
        frames += 1
        proc.append(obs.proc_ms)
        faces += obs.face
        blinks += obs.blink
        downs += obs.looking_down
    el = time.perf_counter() - t0
    cpu = (time.process_time() - c0) / el * 100
    cam.release()
    tracker.close()

    n = max(frames, 1)
    print(f"{el:.0f}秒  {frames / el:.1f}fps  顔検出 {faces / n * 100:.0f}%  瞬き {blinks / n * 100:.1f}%  下を見ている {downs / n * 100:.1f}%")
    print(f"処理時間  平均 {np.mean(proc):.1f}ms  p50 {np.percentile(proc, 50):.1f}ms  p95 {np.percentile(proc, 95):.1f}ms")
    print(f"プロセスのCPU  {cpu:.1f}%(1コア比)  {cpu / (os.cpu_count() or 1):.2f}%(全体、{os.cpu_count()}スレッド)")


if __name__ == "__main__":
    main()
