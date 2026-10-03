"""EyeTraxのGazeEstimatorのラッパー。特徴抽出、瞬きの判定、頭の位置の取得、視線の予測"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gaze_overlay  # noqa: F401  環境変数の設定を cv2/mediapipe より先に済ませるため
import numpy as np

from gaze_overlay.config import TrackerConfig

log = logging.getLogger(__name__)

# EyeTraxの瞬き判定と同じランドマーク番号(外側の目尻、内側の目尻、上まぶた、下まぶた)
_LEFT_EYE = (33, 133, 159, 145)
_RIGHT_EYE = (263, 362, 386, 374)


@dataclass
class HeadPose:
    """カメラ画像内の頭の位置。どれも画像の幅を1とした値"""

    cx: float
    cy: float
    # 両目の外側の目尻の間隔(カメラからの距離の目安)
    iod: float

    def to_list(self) -> list[float]:
        return [self.cx, self.cy, self.iod]


@dataclass
class Observation:
    t: float
    face: bool
    features: np.ndarray | None = None
    ear: float | None = None
    ear_baseline: float | None = None
    # 目を閉じている(瞬き)と判定した
    blink: bool = False
    # まぶたは下がっているが、画面の下を見ているとみなして追跡を続けている
    looking_down: bool = False
    head: HeadPose | None = None
    proc_ms: float = 0.0


class _LandmarkTap:
    """FaceLandmarkerを包み、detect_for_videoの結果(生のランドマーク)を横取りする。
    eyetraxのコードを複製せずに頭の位置とEARを得るため。eyetrax 0.4.0 の private属性に依存する"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.last_result = None

    def detect_for_video(self, image, timestamp_ms):
        result = self._inner.detect_for_video(image, timestamp_ms)
        self.last_result = result
        return result

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _eye_ratio(lms, idx: tuple[int, int, int, int], w: int, h: int) -> float:
    outer, inner, top, bottom = (np.array([lms[i].x * w, lms[i].y * h]) for i in idx)
    width = np.linalg.norm(outer - inner)
    height = np.linalg.norm(top - bottom)
    return float(height / (width + 1e-9))


class GazeTracker:
    def __init__(self, cfg: TrackerConfig, face_model: Path | None) -> None:
        from eyetrax import GazeEstimator

        self.cfg = cfg
        model_path = str(face_model) if face_model and face_model.exists() else None
        if face_model and model_path is None:
            log.warning("顔ランドマークのモデル %s がないため、eyetraxのキャッシュを使います", face_model)
        self.est = GazeEstimator(
            model_name=cfg.model,
            model_kwargs=self._model_kwargs(),
            face_landmarker_model=model_path,
            blink_threshold_ratio=cfg.blink_threshold_ratio,
        )
        inner = getattr(self.est, "_face_landmarker", None)
        if inner is not None and hasattr(inner, "detect_for_video"):
            self._tap: _LandmarkTap | None = _LandmarkTap(inner)
            self.est._face_landmarker = self._tap
        else:
            # eyetraxの内部構造が変わった場合は、瞬き判定をeyetraxに任せ、頭の位置ずれ機能を無効にする
            self._tap = None
            log.warning("ランドマークを取得できないため、頭の位置ずれの検出を無効にします")
        self.trained = False
        self._ear_hist: deque[float] = deque(maxlen=150)
        self._low_since: float | None = None

    def _model_kwargs(self) -> dict | None:
        return {"alpha": self.cfg.ridge_alpha} if self.cfg.model == "ridge" else None

    @property
    def head_available(self) -> bool:
        return self._tap is not None

    def process(self, frame: np.ndarray) -> Observation:
        t = time.monotonic()
        t0 = time.perf_counter()
        if self._tap is not None:
            self._tap.last_result = None
        features, eyetrax_blink = self.est.extract_features(frame)
        if features is None:
            self._low_since = None
            return Observation(t=t, face=False, proc_ms=(time.perf_counter() - t0) * 1000)

        obs = Observation(t=t, face=True, features=features)
        result = self._tap.last_result if self._tap is not None else None
        if result is not None and result.face_landmarks:
            h, w = frame.shape[:2]
            lms = result.face_landmarks[0]
            obs.ear = (_eye_ratio(lms, _LEFT_EYE, w, h) + _eye_ratio(lms, _RIGHT_EYE, w, h)) / 2
            lx, ly = lms[33].x * w, lms[33].y * h
            rx, ry = lms[263].x * w, lms[263].y * h
            obs.head = HeadPose(
                cx=(lx + rx) / 2 / w, cy=(ly + ry) / 2 / w, iod=float(np.hypot(rx - lx, ry - ly)) / w
            )
            self._judge_blink(obs)
        else:
            obs.blink = bool(eyetrax_blink)
        obs.proc_ms = (time.perf_counter() - t0) * 1000
        return obs

    def _judge_blink(self, obs: Observation) -> None:
        ear = obs.ear
        assert ear is not None
        self._ear_hist.append(ear)
        if len(self._ear_hist) >= 15:
            # 開いているときの基準値。瞬きや下を見ている時間に引きずられにくいよう、80パーセンタイルを使う
            baseline = float(np.percentile(self._ear_hist, 80))
            obs.ear_baseline = baseline
            low = ear < baseline * self.cfg.blink_threshold_ratio
            closed = ear < baseline * self.cfg.closed_floor_ratio
        else:
            low = closed = ear < 0.15
        if not low:
            self._low_since = None
            return
        if self._low_since is None:
            self._low_since = obs.t
        long_low = (obs.t - self._low_since) * 1000 >= self.cfg.long_blink_ms
        if long_low and not closed:
            obs.looking_down = True
        else:
            obs.blink = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """キャリブレーションしたモニター内のpx座標を返す"""
        return np.asarray(self.est.predict(features[None, :])[0], dtype=np.float64)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """新しいモデルを学習してから差し替える(カメラスレッドの予測と競合しないように)"""
        from eyetrax.models import create_model

        model = create_model(self.cfg.model, **(self._model_kwargs() or {}))
        model.train(np.asarray(X), np.asarray(y))
        self.est.model = model
        self.trained = True

    def predict_with(self, X: np.ndarray) -> np.ndarray:
        """まとめて予測する(キャリブレーションの検証用)"""
        return np.asarray(self.est.predict(np.asarray(X)), dtype=np.float64)

    def close(self) -> None:
        self.est.close()
