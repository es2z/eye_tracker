"""視線オーバーレイの本体。スレッドを起動し、カメラ → 推論 → 平滑化 → WebSocket配信 をつなぐ

スレッド構成:
- メイン: tkinter(キャリブレーション画面、マーカー、デバッグ表示)とコマンドの処理
- camera: カメラの読み込み、特徴抽出、視線の計算(MediaPipeはこのスレッドだけで呼ぶ)
- targets: ゲームウィンドウの追跡
- server: HTTP + WebSocket(asyncio)
- hotkeys / tray / watch-pid
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gaze_overlay  # noqa: F401  環境変数の設定を cv2/mediapipe より先に済ませるため
import numpy as np

from gaze_overlay import win32
from gaze_overlay.calibration import session as calib
from gaze_overlay.calibration.store import CalibrationData, apply_correction
from gaze_overlay.calibration.ui import FullscreenSurface, OverlaySurface
from gaze_overlay.camera import Camera
from gaze_overlay.config import LOG_DIR, Config, load_config, load_state, save_state
from gaze_overlay.server import Hub
from gaze_overlay.smoothing import OneEuroFilter
from gaze_overlay.targets import TargetTracker
from gaze_overlay.tracker import GazeTracker, Observation

log = logging.getLogger("gaze_overlay")

MUTEX_NAME = "Local\\GazeOverlay"
PROTOCOL_VERSION = 1


@dataclass
class TrackState:
    face_streak: int = 0
    lost_since: float | None = None
    skip_frames: int = 0


class App:
    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.state_store = load_state()
        presets = cfg.smoothing.presets
        preset = self.state_store.get("preset", cfg.smoothing.preset)
        self.preset = preset if preset in presets else next(iter(presets))
        self.shown: bool = bool(self.state_store.get("shown", True))
        self.paused = False
        self.calibrating: str | None = None
        self.commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.tracker_ready = threading.Event()

        self.calib_monitor = win32.find_monitor(cfg.calibration.monitor)
        self.calib_path = cfg.resolve(cfg.calibration.file)
        self.calib_data = CalibrationData.load(self.calib_path)

        self.camera = Camera(cfg.camera)
        self.tracker: GazeTracker | None = None
        self.collector = calib.Collector()
        self.filter = OneEuroFilter(**presets[self.preset])
        self.targets = TargetTracker(cfg.target)
        mode = args.mode or self.state_store.get("mode")
        if mode in ("window", "monitor"):
            self.targets.mode = mode
        self.targets.on_change(self._on_target_change)

        self.hub = Hub(cfg.server, self._on_remote_command, self._hello, self._status)
        self.hotkeys = None
        self.tray = None

        # カメラスレッドが更新し、他のスレッドが読む値(代入は原子的なのでロックは使わない)
        self.track = TrackState()
        self.gaze: np.ndarray | None = None
        self.raw_gaze: np.ndarray | None = None
        self.state = "starting"
        self.last_obs: Observation | None = None
        self.head_drift: dict | None = None
        self._seq = 0
        self._fps = 0.0
        self._proc_ms = 0.0
        self._cpu_pct = 0.0
        self._idle = False

        self.root: tk.Tk | None = None
        self.overlay_surface: OverlaySurface | None = None
        self._sequence: calib.PointSequence | None = None
        self._last_tray = 0.0
        self._camera_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ 起動と終了

    def run(self) -> int:
        if not self.hub.start():
            return 1
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.report_callback_exception = lambda *exc: log.error("UIでエラー", exc_info=exc)
        self.overlay_surface = OverlaySurface(self.root, self.cfg.ui.font)

        self._camera_thread = threading.Thread(target=self._camera_loop, name="camera", daemon=True)
        self._camera_thread.start()
        threading.Thread(target=self.targets.run, args=(self.stop_event,), name="targets", daemon=True).start()

        if self.cfg.hotkeys.enabled and not self.args.no_hotkeys:
            from gaze_overlay.hotkeys import HotkeyManager

            self.hotkeys = HotkeyManager(self.cfg.hotkeys.bindings, lambda name: self.commands.put((name, None)))
            self.hotkeys.start()
        if self.cfg.ui.tray and not self.args.no_tray:
            try:
                from gaze_overlay.tray import Tray

                self.tray = Tray(lambda name, value: self.commands.put((name, value)), self._tray_state)
                self.tray.start()
            except Exception:
                log.exception("トレイアイコンを作れませんでした")
        if self.args.watch_pid:
            win32.watch_process(self.args.watch_pid, lambda: self.commands.put(("shutdown", "parent-exit")))

        self.root.after(15, self._poll)
        log.info("起動しました。オーバーレイ: %s/overlay  ドック: %s/dock", self.hub.url, self.hub.url)
        try:
            self.root.mainloop()
        finally:
            self._shutdown()
        return 0

    def _shutdown(self) -> None:
        log.info("終了します")
        self.stop_event.set()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        if self.tray is not None:
            self.tray.stop()
        self.hub.stop()
        # カメラを確実に解放するため、カメラスレッドの後片付けを待つ
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3)

    # ------------------------------------------------------------------ カメラスレッド

    def _camera_loop(self) -> None:
        try:
            self.tracker = GazeTracker(self.cfg.tracker, self.cfg.resolve(self.cfg.tracker.face_model))
        except Exception:
            log.exception("視線推定の初期化に失敗しました")
            self.state = "error"
            return
        if self.calib_data is not None:
            try:
                self.tracker.train(self.calib_data.X, self.calib_data.y)
                log.info("保存したキャリブレーションを読み込みました (%d サンプル)", len(self.calib_data.X))
            except Exception:
                log.exception("保存したキャリブレーションで学習できませんでした")
                self.calib_data = None
        self.tracker_ready.set()
        if not self.camera.open():
            log.error("カメラを開けませんでした。%.0f秒ごとに再試行します", self.cfg.camera.reopen_interval_s)
        if self.args.calibrate or (self.calib_data is None and self.cfg.calibration.auto_calibrate_if_missing):
            self.commands.put(("calibrate_full", "auto"))

        checked_dim = False
        frames = 0
        proc_sum = 0.0
        win_start = time.perf_counter()
        cpu_start = time.process_time()
        last_frame_t = 0.0
        try:
            while not self.stop_event.is_set():
                frame = self.camera.read()
                if frame is None:
                    if not self.camera.is_open:
                        self.state = "no_camera"
                        self._publish_gaze()
                    continue
                now = time.monotonic()
                # 一時停止中は推論しない。カメラは開いたままにして、すぐ再開できるようにする
                if self.paused:
                    self.state = "paused"
                    self._publish_gaze()
                    time.sleep(0.1)
                    continue
                # 見ている人がいないとき(オーバーレイが非表示・未接続)は推論の頻度を下げる
                self._idle = (
                    self.hub.active_overlay_count() == 0 and self.calibrating is None and not self.args.debug_view
                )
                if self._idle and now - last_frame_t < 1.0 / max(self.cfg.tracker.idle_fps, 0.1):
                    continue
                last_frame_t = now

                obs = self.tracker.process(frame)
                self.last_obs = obs
                if not checked_dim and obs.features is not None and self.calib_data is not None:
                    checked_dim = True
                    if self.calib_data.X.shape[1] != obs.features.shape[0]:
                        log.error("保存したキャリブレーションの特徴量の次元が合いません。やり直してください")
                        self.calib_data = None
                        self.tracker.trained = False
                        self.commands.put(("calibrate_full", "auto"))
                self.collector.observe(obs)
                self._update_gaze(obs)
                self._update_drift(obs)
                self._publish_gaze()

                frames += 1
                proc_sum += obs.proc_ms
                elapsed = time.perf_counter() - win_start
                if elapsed >= 2.0:
                    # CPU使用率はシステム全体ではなく、このプロセス自身のCPU時間で測る
                    self._cpu_pct = (time.process_time() - cpu_start) / elapsed * 100
                    self._fps = frames / elapsed
                    self._proc_ms = proc_sum / max(frames, 1)
                    frames, proc_sum = 0, 0.0
                    win_start = time.perf_counter()
                    cpu_start = time.process_time()
        except Exception:
            log.exception("カメラスレッドでエラー")
            self.state = "error"
        finally:
            self.camera.release()
            if self.tracker is not None:
                self.tracker.close()

    def _update_gaze(self, obs: Observation) -> None:
        st = self.track
        tc = self.cfg.tracker
        data = self.calib_data  # メインスレッドが差し替えても、このフレームの中では同じものを使う
        if self.tracker is None or not self.tracker.trained or data is None:
            self.state = "uncalibrated" if obs.face else "lost"
            return
        if not obs.face:
            st.face_streak = 0
            if st.lost_since is None:
                st.lost_since = obs.t
            self.state = "lost"
            return
        st.face_streak += 1
        if st.lost_since is not None:
            if st.face_streak < tc.reacquire_frames:
                self.state = "lost"
                return
            if (obs.t - st.lost_since) * 1000 >= tc.reset_after_lost_ms:
                self.filter.reset()
            st.lost_since = None
        if obs.blink:
            st.skip_frames = tc.post_blink_skip_frames
            self.state = "blink"
            return
        self.state = "tracking"
        if st.skip_frames > 0:
            # 瞬きの直後は位置を更新しない(表示は戻す)
            st.skip_frames -= 1
            return
        assert obs.features is not None
        raw = self.tracker.predict(obs.features) + data.origin
        corrected = apply_correction(data.correction, raw)
        self.raw_gaze = corrected
        self.gaze = self.filter(corrected, obs.t)

    def _update_drift(self, obs: Observation) -> None:
        ref = self.calib_data.head_ref if self.calib_data is not None else None
        if obs.head is None or not ref:
            return
        rcx, rcy, riod = ref
        shift = float(np.hypot(obs.head.cx - rcx, obs.head.cy - rcy) / max(riod, 1e-6))
        scale = abs(obs.head.iod / max(riod, 1e-6) - 1.0)
        prev = self.head_drift
        a = 0.1  # ノイズで警告が点滅しないよう指数移動平均をとる
        if prev is not None:
            shift = prev["shift"] * (1 - a) + shift * a
            scale = prev["scale"] * (1 - a) + scale * a
        tc = self.cfg.tracker
        score = max(shift / tc.drift_shift_warn, scale / tc.drift_scale_warn)
        self.head_drift = {"shift": shift, "scale": scale, "score": score, "warn": score >= 1.0}

    def _publish_gaze(self) -> None:
        target = self.targets.current
        state = self.state
        if self.calibrating is not None:
            state = "calibrating"
        elif self.paused:
            state = "paused"
        u = v = None
        if target is not None and self.gaze is not None:
            x, y, w, h = target.rect
            u = (float(self.gaze[0]) - x) / w
            v = (float(self.gaze[1]) - y) / h
        visible = state == "tracking" and self.shown and target is not None and u is not None
        self._seq += 1
        self.hub.publish(
            "overlay",
            {
                "type": "gaze",
                "seq": self._seq,
                "t": time.time() * 1000,
                "u": u,
                "v": v,
                "visible": visible,
                "state": state if target is not None else "no_target",
            },
        )

    def _on_target_change(self, target) -> None:
        self.hub.publish("overlay", {"type": "target", "target": target.to_dict() if target else None})

    # ------------------------------------------------------------------ WebSocket向けの情報

    def _hello(self, role: str) -> dict:
        target = self.targets.current
        return {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "role": role,
            "target": target.to_dict() if target else None,
        }

    def _status(self) -> dict:
        target = self.targets.current
        obs = self.last_obs
        data = self.calib_data
        calib_info: dict[str, Any] = {"exists": data is not None}
        if data is not None:
            calib_info.update(
                {
                    "age_s": time.time() - data.created,
                    "samples": len(data.X),
                    "error_px": (data.validation or {}).get("error_px"),
                    "error_pct": (data.validation or {}).get("error_pct"),
                    "correction": data.correction_kind,
                }
            )
        return {
            "type": "status",
            "state": "calibrating" if self.calibrating else ("paused" if self.paused else self.state),
            "calibrating": self.calibrating,
            "fps": round(self._fps, 1),
            "proc_ms": round(self._proc_ms, 1),
            "cpu_pct": round(self._cpu_pct, 1),
            "cpu_total_pct": round(self._cpu_pct / (os.cpu_count() or 1), 2),
            "idle": self._idle,
            "shown": self.shown,
            "paused": self.paused,
            "preset": self.preset,
            "presets": list(self.cfg.smoothing.presets),
            "mode": self.targets.mode,
            "target": target.to_dict() if target else None,
            "camera": {**self.camera.info.to_dict(), "open": self.camera.is_open},
            "calib": calib_info,
            "head_drift": self.head_drift,
            "head_available": bool(self.tracker and self.tracker.head_available),
            "ear": round(obs.ear, 3) if obs and obs.ear is not None else None,
            "ear_baseline": round(obs.ear_baseline, 3) if obs and obs.ear_baseline is not None else None,
            "looking_down": bool(obs and obs.looking_down),
            "hotkeys": self.hotkeys.results if self.hotkeys else {},
            "overlays": [{"viewport": c.viewport, "visible": c.visible} for c in self.hub.overlays()],
        }

    def _tray_state(self) -> dict:
        return {
            "shown": self.shown,
            "paused": self.paused,
            "preset": self.preset,
            "presets": list(self.cfg.smoothing.presets),
            "mode": self.targets.mode,
        }

    def _on_remote_command(self, name: str, value: Any) -> None:
        self.commands.put((name, value))

    # ------------------------------------------------------------------ メインスレッド

    def _poll(self) -> None:
        assert self.root is not None
        try:
            while True:
                name, value = self.commands.get_nowait()
                try:
                    self._handle(name, value)
                except Exception:
                    log.exception("コマンド %s の処理でエラー", name)
        except queue.Empty:
            pass
        if self.args.debug_view and self.calibrating is None:
            self._draw_debug()
        now = time.monotonic()
        if self.tray is not None and now - self._last_tray >= 1.0:
            self._last_tray = now
            self._refresh_tray()
        self.root.after(15, self._poll)

    def _handle(self, name: str, value: Any) -> None:
        assert self.root is not None
        if name == "shutdown":
            log.info("終了の要求を受け取りました (%s)", value or "")
            # 終了時の確認ダイアログが必要なら、次の2行のコメントを外す(OBS連動時は表示されないよう value で判定)
            # from tkinter import messagebox
            # if value is None and not messagebox.askyesno("視線オーバーレイ", "終了しますか？"): return
            if self._sequence is not None:
                self._sequence.cancel()
            self.root.quit()
        elif name == "toggle_visible":
            self.shown = not self.shown
            self._save_state()
            log.info("表示: %s", "オン" if self.shown else "オフ")
        elif name == "pause":
            self.paused = not self.paused
            if not self.paused:
                self.filter.reset()
            log.info("一時停止: %s", "オン" if self.paused else "オフ")
        elif name in ("calibrate_full", "calibrate_quick", "offset"):
            self._start_calibration(name, value)
        elif name == "set_preset":
            self._set_preset(str(value))
        elif name == "cycle_preset":
            names = list(self.cfg.smoothing.presets)
            self._set_preset(names[(names.index(self.preset) + 1) % len(names)])
        elif name == "pick_foreground":
            # ドックやトレイから押すと、その時点ではOBSやタスクバーが前面になっている。
            # その場合は value に秒数を渡し、対象のゲームへ切り替える時間を待ってから固定する
            delay = float(value or 0)
            if delay > 0:
                self._notify(f"{delay:.0f}秒後の前面ウィンドウを対象にします。ゲームをクリックしてください")
                self.root.after(int(delay * 1000), lambda: self._handle("pick_foreground", None))
                return
            self.targets.pin_foreground()
            self._save_state()
        elif name == "set_mode":
            self.targets.set_mode(str(value))
            self._save_state()
        elif name == "copy_dock_url":
            self.root.clipboard_clear()
            self.root.clipboard_append(f"{self.hub.url}/dock")
        elif name == "open_logs":
            os.startfile(LOG_DIR)
        else:
            log.warning("不明なコマンド: %s", name)

    def _set_preset(self, name: str) -> None:
        params = self.cfg.smoothing.presets.get(name)
        if params is None:
            log.warning("不明なプリセット: %s", name)
            return
        self.preset = name
        self.filter.configure(**params)
        self._save_state()
        log.info("平滑化のプリセット: %s", name)

    def _save_state(self) -> None:
        self.state_store.update({"preset": self.preset, "shown": self.shown, "mode": self.targets.mode})
        save_state(self.state_store)

    def _notify(self, message: str) -> None:
        log.info(message)
        if self.tray is not None:
            self.tray.notify(message)

    # ------------------------------------------------------------------ キャリブレーション

    def _start_calibration(self, kind: str, value: Any) -> None:
        assert self.root is not None and self.overlay_surface is not None
        if self._sequence is not None:
            # 実行中にもう一度ホットキーを押したら中止する
            log.info("キャリブレーションを中止します")
            self._sequence.cancel()
            return
        if not self.tracker_ready.is_set() or self.tracker is None:
            # 視線推定の初期化が終わるまで待つ
            self.root.after(500, lambda: self.commands.put((kind, value)))
            return
        if kind != "calibrate_full" and (self.calib_data is None or not self.tracker.trained):
            self._notify("先にフルキャリブレーションを行ってください")
            return
        cc = self.cfg.calibration
        self.calibrating = kind
        if kind == "calibrate_full":
            self.calib_monitor = win32.find_monitor(cc.monitor)
            mon = self.calib_monitor
            points = calib.full_points(mon.rect, cc.margin, cc.pulse_s, cc.capture_s, cc.validate, cc.validate_capture_s)
            surface = FullscreenSurface(self.root, self.cfg.ui.font, mon.rect, on_escape=lambda: self._cancel_sequence())
            self._sequence = calib.PointSequence(
                self.root, surface, self.collector, points,
                on_done=lambda s, h: self._finish_full(surface, points, mon, s, h),
                wait_face=True,
                intro="点が出たら、点を見つめてください",
            )
            log.info("フルキャリブレーションを開始します (%s %s)", mon.name, mon.rect)
        else:
            target = self.targets.current
            rect = target.rect if target is not None else self.calib_monitor.rect
            if kind == "calibrate_quick":
                points = calib.quick_points(rect, cc.quick_margin, cc.quick_settle_s, cc.quick_capture_s)
            else:
                points = calib.offset_point(rect, cc.offset_settle_s, cc.offset_capture_s)
            surface = self.overlay_surface
            surface.cover(rect)
            self._sequence = calib.PointSequence(
                self.root, surface, self.collector, points,
                on_done=lambda s, h: self._finish_correction(kind, points, rect, s),
            )
            log.info("%sを開始します %s", "クイック補正" if kind == "calibrate_quick" else "1点補正", rect)
        self._sequence.start()

    def _cancel_sequence(self) -> None:
        if self._sequence is not None:
            self._sequence.cancel()

    def _finish_full(self, surface: FullscreenSurface, points, mon: win32.Monitor, samples, heads) -> None:
        surface.close()
        self._sequence = None
        self.calibrating = None
        if samples is None:
            log.info("フルキャリブレーションを中止しました")
            return
        from eyetrax import __version__ as eyetrax_version

        assert self.tracker is not None
        info = self.camera.info
        data = calib.finish_full(
            self.tracker, samples, heads, points, mon.to_dict(),
            [info.process_width, info.process_height], eyetrax_version,
        )
        if data is None:
            self._notify("キャリブレーションに失敗しました(顔を検出できた点が足りません)")
            return
        data.save(self.calib_path)
        self.calib_data = data
        self.head_drift = None
        self.filter.reset()
        if data.validation:
            v = data.validation
            self._notify(f"キャリブレーション完了: 平均誤差 {v['error_px']:.0f}px(画面の高さの{v['error_pct']:.1f}%)")
        else:
            self._notify("キャリブレーション完了")

    def _finish_correction(self, kind: str, points, rect, samples) -> None:
        assert self.overlay_surface is not None and self.tracker is not None
        self._sequence = None
        self.calibrating = None
        if self.args.debug_view:
            self.overlay_surface.cover(self.calib_monitor.rect)
        else:
            self.overlay_surface.shrink()
        if samples is None or self.calib_data is None:
            log.info("補正を中止しました")
            return
        if kind == "calibrate_quick":
            ok = calib.finish_quick(self.tracker, self.calib_data, samples, points, rect)
        else:
            ok = calib.finish_offset(self.tracker, self.calib_data, samples, points)
        if ok:
            self.calib_data.save(self.calib_path)
            self.filter.reset()
            self._notify("補正しました")
        else:
            self._notify("補正に失敗しました(顔を検出できませんでした)")

    # ------------------------------------------------------------------ 表示の補助

    def _draw_debug(self) -> None:
        surface = self.overlay_surface
        assert surface is not None
        surface.cover(self.calib_monitor.rect)
        surface.clear()
        obs = self.last_obs
        drift = self.head_drift
        lines = [
            f"{self.state}  {self._fps:.0f}fps  処理 {self._proc_ms:.1f}ms  CPU {self._cpu_pct:.0f}%(1コア比)",
        ]
        if obs is not None and obs.ear is not None:
            base = f"{obs.ear_baseline:.3f}" if obs.ear_baseline else "-"
            lines.append(f"EAR {obs.ear:.3f} / 基準 {base}{'  下を見ている' if obs.looking_down else ''}")
        if drift is not None:
            lines.append(f"頭のずれ {drift['score']:.2f}{'  再補正推奨' if drift['warn'] else ''}")
        target = self.targets.current
        lines.append(f"対象: {target.mode} {target.exe} {target.rect}" if target else "対象: なし")
        x0, y0 = surface.origin
        surface.canvas.create_text(
            20, 20, text="\n".join(lines), anchor="nw", fill="#ffffff", font=(self.cfg.ui.font, 13, "bold")
        )
        if target is not None:
            tx, ty, tw, th = target.rect
            surface.canvas.create_rectangle(tx - x0, ty - y0, tx - x0 + tw, ty - y0 + th, outline="#00c8ff", width=2)
        if self.gaze is not None and self.state == "tracking":
            raw = tuple(self.raw_gaze) if self.raw_gaze is not None else None
            surface.draw_cursor(float(self.gaze[0]), float(self.gaze[1]), raw, "")

    def _refresh_tray(self) -> None:
        assert self.tray is not None
        drift = self.head_drift
        if self.calib_data is None:
            kind, title = "warn", "視線オーバーレイ: 未キャリブレーション"
        elif self.paused:
            kind, title = "off", "視線オーバーレイ: 一時停止中"
        elif drift is not None and drift["warn"]:
            kind, title = "warn", "視線オーバーレイ: 頭の位置がずれています(再補正推奨)"
        elif self.targets.current is None:
            # ゲームウィンドウ追従で対象がないときはバブルを出さないので、理由をツールチップに出す
            kind, title = "off", "視線オーバーレイ: 対象のゲームなし(ゲームを前面にするか、モニター全体に切り替え)"
        else:
            kind = "ok" if self.shown else "off"
            title = f"視線オーバーレイ: {self.state} {self._fps:.0f}fps"
        self.tray.refresh(title, kind)


# ---------------------------------------------------------------------- エントリーポイント

def _setup_logging(level: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level.upper())
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "gaze_overlay.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # websocketsの接続ごとのログは多いので警告以上だけにする(接続と切断は server.py が記録する)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    # OBSからコンソールなしで起動したときは標準エラー出力がないことがある
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _send_stop(host: str, port: int) -> int:
    """起動中のインスタンスに終了を伝える(OBSのLuaスクリプトの停止ボタンから使う)"""
    import json

    from websockets.sync.client import connect

    try:
        with connect(f"ws://{host}:{port}/ws", open_timeout=3) as ws:
            ws.send(json.dumps({"type": "hello", "role": "control"}))
            ws.send(json.dumps({"type": "cmd", "name": "shutdown", "value": "stop-command"}))
        print("終了を要求しました")
        return 0
    except OSError:
        print("起動中のインスタンスが見つかりません")
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="gaze-overlay", description="Webカメラの視線をOBSにバブルとして重ねる")
    p.add_argument("--config", type=Path, help="設定ファイル(既定: プロジェクト直下の config.toml)")
    p.add_argument("--debug-view", action="store_true", help="視線カーソルと状態をデスクトップに表示する(確認用)")
    p.add_argument("--calibrate", action="store_true", help="起動時にフルキャリブレーションを行う")
    p.add_argument("--mode", choices=["window", "monitor"], help="座標の基準(既定は設定ファイルの値)")
    p.add_argument("--watch-pid", type=int, help="このプロセス(OBS)が終了したら一緒に終了する")
    p.add_argument("--stop", action="store_true", help="起動中のインスタンスを終了させる")
    p.add_argument("--port", type=int, help="ポート番号(既定は設定ファイルの値)")
    p.add_argument("--no-tray", action="store_true", help="トレイアイコンを出さない")
    p.add_argument("--no-hotkeys", action="store_true", help="グローバルホットキーを登録しない")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.port:
        cfg.server.port = args.port
    if args.stop:
        sys.exit(_send_stop(cfg.server.host, cfg.server.port))

    win32.set_dpi_awareness()
    _setup_logging(cfg.log.level)
    instance = win32.SingleInstance(MUTEX_NAME)
    if instance.already_running:
        log.info("すでに起動しているため終了します")
        sys.exit(0)
    # 自動起動の直後はカメラやVTuberアプリの準備が終わっていないことがある。待つ場合は次の行のコメントを外す
    # time.sleep(5)
    # ゲームの処理を優先させたい場合は、次の2行のコメントを外してプロセスの優先度を下げる
    # (CPUが混んでいるとフレームが遅れることがあるため、既定では通常の優先度にしている)
    # import ctypes
    # ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    try:
        code = App(cfg, args).run()
    finally:
        instance.close()
    sys.exit(code)
