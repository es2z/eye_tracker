"""キャリブレーションの進行と結果の計算

- Collector: カメラスレッドから特徴量を受け取り、表示中の点ごとに集める(スレッドセーフ)
- PointSequence: tkinterの after() で点を順番に表示する(メインスレッド)
- finish_*: 集めたサンプルから、モデルの学習や補正行列を計算する
"""

from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from gaze_overlay.calibration.store import CalibrationData, apply_correction, fit_correction, with_offset
from gaze_overlay.calibration.ui import Surface
from gaze_overlay.tracker import GazeTracker, Observation

log = logging.getLogger(__name__)

# 1点あたりの最低サンプル数(これより少ない点は捨てる)
MIN_SAMPLES = 5


@dataclass
class PointSpec:
    # 仮想デスクトップ上の座標 [px]
    gx: float
    gy: float
    settle_s: float
    capture_s: float
    # "train"(学習用)または "validate"(検証用)
    kind: str = "train"


class Collector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: int | None = None
        self._samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._heads: list[list[float]] = []
        self._face_since: float | None = None

    def observe(self, obs: Observation) -> None:
        """カメラスレッドから毎フレーム呼ぶ"""
        usable = obs.face and not obs.blink and obs.features is not None
        with self._lock:
            if usable:
                if self._face_since is None:
                    self._face_since = obs.t
            else:
                self._face_since = None
            if self._active is not None and usable:
                self._samples[self._active].append(obs.features)
                if obs.head is not None:
                    self._heads.append(obs.head.to_list())

    def face_ok_for(self) -> float:
        """顔を(瞬きせずに)連続で検出している秒数"""
        with self._lock:
            return 0.0 if self._face_since is None else time.monotonic() - self._face_since

    def begin(self, index: int) -> None:
        with self._lock:
            self._active = index

    def end(self) -> None:
        with self._lock:
            self._active = None

    def take(self) -> tuple[dict[int, list[np.ndarray]], list[list[float]]]:
        with self._lock:
            samples, heads = dict(self._samples), list(self._heads)
            self._samples = defaultdict(list)
            self._heads = []
            self._active = None
            return samples, heads


class PointSequence:
    """点を順番に表示し、収集中の点を Collector に伝える"""

    def __init__(
        self,
        root: tk.Tk,
        surface: Surface,
        collector: Collector,
        points: list[PointSpec],
        on_done: Callable[[dict[int, list[np.ndarray]] | None, list[list[float]]], None],
        wait_face: bool = False,
        intro: str = "",
    ) -> None:
        self.root = root
        self.surface = surface
        self.collector = collector
        self.points = points
        self.on_done = on_done
        self.wait_face = wait_face
        self.intro = intro
        self._state = "wait_face" if wait_face else "point"
        self._index = 0
        self._phase = "settle"
        self._phase_start = 0.0
        self._after_id: str | None = None
        self._finished = False

    def start(self) -> None:
        self.collector.take()
        self._phase_start = time.monotonic()
        self._tick()

    def cancel(self) -> None:
        if self._finished:
            return
        self._finish(cancelled=True)

    def _finish(self, cancelled: bool) -> None:
        self._finished = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.collector.end()
        samples, heads = self.collector.take()
        self.on_done(None if cancelled else samples, heads)

    def _tick(self) -> None:
        if self._finished:
            return
        now = time.monotonic()
        s = self.surface
        s.clear()
        if self._state == "wait_face":
            ok_for = self.collector.face_ok_for()
            if ok_for >= 1.0:
                self._state = "point"
                self._phase = "settle"
                self._phase_start = now
            else:
                s.draw_message(
                    self.intro or "画面中央の点を見てください",
                    "顔を検出すると始まります(Escで中止)" if ok_for <= 0 else "そのまま…",
                )
        if self._state == "point":
            p = self.points[self._index]
            elapsed = now - self._phase_start
            if self._phase == "settle" and elapsed >= p.settle_s:
                self._phase = "capture"
                self._phase_start = now
                elapsed = 0.0
                self.collector.begin(self._index)
            elif self._phase == "capture" and elapsed >= p.capture_s:
                self.collector.end()
                self._index += 1
                if self._index >= len(self.points):
                    self._finish(cancelled=False)
                    return
                p = self.points[self._index]
                self._phase = "settle"
                self._phase_start = now
                elapsed = 0.0
            duration = p.settle_s if self._phase == "settle" else p.capture_s
            s.draw_target(p.gx, p.gy, self._phase, min(elapsed / max(duration, 1e-6), 1.0), elapsed)
        self._after_id = self.root.after(16, self._tick)


# --- 点の配置 ---

def full_points(
    rect: tuple[int, int, int, int], margin: float, pulse_s: float, capture_s: float,
    validate: bool, validate_capture_s: float,
) -> list[PointSpec]:
    """9点(eyetraxと同じ順番)と、検証用の4点"""
    x, y, w, h = rect
    order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2)]
    mx, my = w * margin, h * margin
    pts = [
        PointSpec(x + mx + c * (w - 2 * mx) / 2, y + my + r * (h - 2 * my) / 2, pulse_s, capture_s)
        for r, c in order
    ]
    if validate:
        for fx, fy in [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]:
            pts.append(PointSpec(x + fx * w, y + fy * h, 0.8, validate_capture_s, "validate"))
    return pts


def quick_points(rect: tuple[int, int, int, int], margin: float, settle_s: float, capture_s: float) -> list[PointSpec]:
    """対象の中央と、四隅から margin だけ内側の4点"""
    x, y, w, h = rect
    fr = [(0.5, 0.5), (margin, margin), (1 - margin, margin), (1 - margin, 1 - margin), (margin, 1 - margin)]
    return [PointSpec(x + fx * w, y + fy * h, settle_s, capture_s) for fx, fy in fr]


def offset_point(rect: tuple[int, int, int, int], settle_s: float, capture_s: float) -> list[PointSpec]:
    x, y, w, h = rect
    return [PointSpec(x + w / 2, y + h / 2, settle_s, capture_s)]


# --- 結果の計算 ---

def _global_predictions(tracker: GazeTracker, feats: list[np.ndarray], origin: np.ndarray) -> np.ndarray:
    return tracker.predict_with(np.asarray(feats)) + origin


def finish_full(
    tracker: GazeTracker,
    samples: dict[int, list[np.ndarray]],
    heads: list[list[float]],
    points: list[PointSpec],
    monitor: dict,
    camera: list[int],
    eyetrax_version: str,
) -> CalibrationData | None:
    origin = np.array([monitor["x"], monitor["y"]], dtype=np.float64)
    X, Y = [], []
    for i, p in enumerate(points):
        feats = samples.get(i, [])
        if p.kind != "train" or len(feats) < MIN_SAMPLES:
            continue
        X.extend(feats)
        # モデルはモニター内の座標で学習する
        Y.extend([[p.gx - origin[0], p.gy - origin[1]]] * len(feats))
    n_train = sum(1 for i, p in enumerate(points) if p.kind == "train" and len(samples.get(i, [])) >= MIN_SAMPLES)
    if n_train < 5:
        log.warning("有効な点が %d 点しかないため、キャリブレーションを中止します", n_train)
        return None
    X_arr, Y_arr = np.asarray(X), np.asarray(Y, dtype=np.float64)
    tracker.train(X_arr, Y_arr)

    errors = []
    for i, p in enumerate(points):
        feats = samples.get(i, [])
        if p.kind != "validate" or len(feats) < MIN_SAMPLES:
            continue
        pred = np.median(_global_predictions(tracker, feats, origin), axis=0)
        errors.append(float(np.hypot(pred[0] - p.gx, pred[1] - p.gy)))
    validation = None
    if errors:
        err = float(np.mean(errors))
        validation = {"error_px": err, "error_pct": err / monitor["height"] * 100, "points": len(errors)}
        log.info("検証: 平均誤差 %.0fpx (モニターの高さの %.1f%%、%d点)", err, validation["error_pct"], len(errors))

    head_ref = np.median(np.asarray(heads), axis=0).tolist() if heads else None
    return CalibrationData(
        X=X_arr,
        y=Y_arr,
        monitor=monitor,
        camera=camera,
        head_ref=head_ref,
        eyetrax_version=eyetrax_version,
        validation=validation,
    )


def finish_quick(
    tracker: GazeTracker,
    data: CalibrationData,
    samples: dict[int, list[np.ndarray]],
    points: list[PointSpec],
    rect: tuple[int, int, int, int],
) -> bool:
    """基本モデルの予測値(補正前)を目標点へ合わせる補正行列を求め、data の補正を置き換える"""
    P, T = [], []
    for i, p in enumerate(points):
        feats = samples.get(i, [])
        if len(feats) < MIN_SAMPLES:
            continue
        P.append(np.median(_global_predictions(tracker, feats, data.origin), axis=0))
        T.append([p.gx, p.gy])
    if not P:
        log.warning("クイック補正: 有効な点がありませんでした")
        return False
    A, kind = fit_correction(np.asarray(P), np.asarray(T), rect)
    data.correction = A
    data.correction_kind = kind
    data.correction_time = time.time()
    log.info("クイック補正: %s (%d点)", kind, len(P))
    return True


def finish_offset(
    tracker: GazeTracker, data: CalibrationData, samples: dict[int, list[np.ndarray]], points: list[PointSpec]
) -> bool:
    """今の補正を掛けた予測値と目標点の差を、平行移動として補正に足す"""
    feats = samples.get(0, [])
    if len(feats) < MIN_SAMPLES:
        log.warning("1点補正: サンプルが足りませんでした (%d)", len(feats))
        return False
    raw = _global_predictions(tracker, feats, data.origin)
    corrected = np.array([apply_correction(data.correction, p) for p in raw])
    pred = np.median(corrected, axis=0)
    d = np.array([points[0].gx, points[0].gy]) - pred
    data.correction = with_offset(data.correction, d)
    if data.correction_kind in ("none", "offset"):
        data.correction_kind = "offset"
    elif not data.correction_kind.endswith("+offset"):
        data.correction_kind += "+offset"
    data.correction_time = time.time()
    log.info("1点補正: (%.0f, %.0f)px 移動", d[0], d[1])
    return True
