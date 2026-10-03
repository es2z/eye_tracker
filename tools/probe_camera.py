"""カメラの確認

    uv run python tools/probe_camera.py                      # カメラ名の一覧と、config.toml のカメラの確認
    uv run python tools/probe_camera.py --name "4K Webcam"   # 名前を指定して確認
    uv run python tools/probe_camera.py --hold 60            # 60秒開いたままにして、他のアプリへの影響を見る

わかっていること(2026-10-03 に確認):
- Windowsのマルチアプリカメラで共有できるのは Media Foundation のアプリ同士だけ。
  VMagicMirror や OBS の映像キャプチャデバイス(DirectShow)とは共有できない
- NVIDIA Broadcast の仮想カメラは複数のアプリから同時に読めるが、出力解像度(1920x1080)と
  同じ解像度を要求しないと、黒い画像が1秒に1枚しか来ない
"""

from __future__ import annotations

import argparse
import os
import time
import winreg

import gaze_overlay  # noqa: F401  環境変数の設定を cv2 より先に済ませるため
import cv2

from gaze_overlay.camera import list_dshow_devices
from gaze_overlay.config import load_config

RESOLUTIONS = [(1920, 1080), (1280, 720), (640, 480)]


def camera_users() -> list[str]:
    """今カメラ(物理)を使っているアプリ(Windowsのプライバシー設定の記録から)"""
    base = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
    users = []

    def walk(key, path):
        for i in range(winreg.QueryInfoKey(key)[0]):
            name = winreg.EnumKey(key, i)
            with winreg.OpenKey(key, name) as sub:
                try:
                    stop, _ = winreg.QueryValueEx(sub, "LastUsedTimeStop")
                    if stop == 0:
                        users.append(os.path.basename(name.replace("#", "\\")))
                except FileNotFoundError:
                    pass
                walk(sub, f"{path}\\{name}")

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, base) as key:
            walk(key, base)
    except FileNotFoundError:
        pass
    return users


def probe(index: int, backend: int, label: str, res: tuple[int, int] | None, sec: float = 2.0) -> None:
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"  {label} 要求={res}: 開けませんでした")
        return
    t_open = time.perf_counter() - t0
    if res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
    n = 0
    frame = None
    level = 0.0
    t = time.perf_counter()
    c0 = time.process_time()
    while time.perf_counter() - t < sec:
        ok, f = cap.read()
        if ok and f is not None:
            n += 1
            frame = f
            level = float(f[::16, ::16].mean())
    el = time.perf_counter() - t
    cpu = (time.process_time() - c0) / el * 100
    cap.release()
    if frame is None:
        print(f"  {label} 要求={res}: 開けたがフレームが来ない(他のアプリが独占している可能性)")
        return
    h, w = frame.shape[:2]
    note = ""
    if n / el < 5 or level < 8:
        note = "  ← 映像が来ていない(黒い・遅い。仮想カメラなら解像度が出力と違う)"
    print(
        f"  {label} 要求={str(res):12} → 実際 {w}x{h}  {n / el:5.1f}fps  平均輝度 {level:5.1f}  "
        f"開く {t_open:4.2f}秒  CPU {cpu:4.1f}%(1コア比){note}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="DirectShowのカメラ名(既定は config.toml の camera.name)")
    ap.add_argument("--hold", type=float, default=0, help="最後に開いたまま指定秒数待つ")
    args = ap.parse_args()

    cfg = load_config()
    name = args.name if args.name is not None else cfg.camera.name
    users = camera_users()
    print("物理カメラを使用中のアプリ:", ", ".join(users) if users else "なし")
    names = list_dshow_devices()
    print("DirectShowのカメラ:")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")

    if name:
        idx = next((i for i, n in enumerate(names) if n.lower() == name.lower()), None)
        if idx is None:
            print(f"「{name}」が見つかりません")
            return
        print(f"「{name}」(DirectShow {idx}) を確認します")
        for res in RESOLUTIONS:
            probe(idx, cv2.CAP_DSHOW, "DSHOW", res)
        hold = (idx, cv2.CAP_DSHOW, (cfg.camera.width, cfg.camera.height))
    else:
        print(f"カメラ {cfg.camera.index} を確認します")
        for label, be in (("MSMF ", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)):
            for res in [None, *RESOLUTIONS]:
                probe(cfg.camera.index, be, label, res)
        hold = (cfg.camera.index, cv2.CAP_MSMF, (cfg.camera.width, cfg.camera.height))

    if args.hold > 0:
        idx, be, (w, h) = hold
        print(f"{w}x{h} で {args.hold:.0f}秒開いたままにします。他のアプリ(VMagicMirrorなど)が滑らかに動くか確認してください")
        probe(idx, be, "保持 ", (w, h), sec=args.hold)


if __name__ == "__main__":
    main()
