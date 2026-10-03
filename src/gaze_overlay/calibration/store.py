"""キャリブレーションのデータの保存と読み込み、補正行列の計算

モデル(pickle)ではなく生のサンプルを保存し、起動時に学習し直す(数ms)。
pickleはscikit-learnのバージョンが変わると読めなくなるため。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class CalibrationData:
    # 特徴量 (n, d) と、キャリブレーションしたモニター内の目標座標 (n, 2) [px]
    X: np.ndarray
    y: np.ndarray
    monitor: dict
    camera: list[int]
    # キャリブレーション時の頭の位置 [cx, cy, iod](tracker.HeadPose)。取れなかったときは None
    head_ref: list[float] | None = None
    created: float = field(default_factory=time.time)
    eyetrax_version: str = ""
    validation: dict | None = None
    # 予測値(仮想デスクトップのpx)に掛ける2x3の補正行列。クイック補正と1点補正で更新する
    correction: np.ndarray = field(default_factory=lambda: IDENTITY.copy())
    correction_kind: str = "none"
    correction_time: float | None = None

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.monitor["x"], self.monitor["y"]], dtype=np.float64)

    def reset_correction(self) -> None:
        self.correction = IDENTITY.copy()
        self.correction_kind = "none"
        self.correction_time = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "monitor": self.monitor,
            "camera": self.camera,
            "head_ref": self.head_ref,
            "created": self.created,
            "eyetrax_version": self.eyetrax_version,
            "validation": self.validation,
            "correction_kind": self.correction_kind,
            "correction_time": self.correction_time,
        }
        # np.savez は拡張子 .npz を自動で付けるため、一時ファイル名も .npz で終わらせる
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            X=self.X,
            y=self.y,
            correction=self.correction,
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
        )
        tmp.replace(path)
        log.info("キャリブレーションを保存しました: %s (%d サンプル)", path, len(self.X))

    @classmethod
    def load(cls, path: Path) -> "CalibrationData | None":
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                return cls(
                    X=z["X"],
                    y=z["y"],
                    correction=z["correction"],
                    monitor=meta["monitor"],
                    camera=meta["camera"],
                    head_ref=meta.get("head_ref"),
                    created=meta.get("created", 0.0),
                    eyetrax_version=meta.get("eyetrax_version", ""),
                    validation=meta.get("validation"),
                    correction_kind=meta.get("correction_kind", "none"),
                    correction_time=meta.get("correction_time"),
                )
        except Exception:
            log.exception("キャリブレーションを読み込めませんでした: %s", path)
            return None


def apply_correction(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    return A[:, :2] @ p + A[:, 2]


def with_offset(A: np.ndarray, d: np.ndarray) -> np.ndarray:
    """補正の後に平行移動 d を足した行列を返す"""
    B = A.copy()
    B[:, 2] += d
    return B


def fit_correction(
    P: np.ndarray, T: np.ndarray, rect: tuple[float, float, float, float], reg: float = 0.2
) -> tuple[np.ndarray, str]:
    """点ごとの予測値 P (n,2) を目標 T (n,2) に合わせる補正行列(2x3)を求める

    恒等変換に近づくよう正則化した最小二乗でaffine変換を求める(平行移動は正則化しない)。
    点が少ない、または結果が極端(拡大縮小が0.5〜2倍の範囲外)なら、平行移動だけにする。
    rect は数値を安定させるための正規化に使う(対象の中心と高さ)。
    """
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n = len(P)
    if n == 0:
        return IDENTITY.copy(), "none"
    offset = with_offset(IDENTITY, (T - P).mean(axis=0))
    if n < 3:
        return offset, "offset"

    x, y, w, h = rect
    c = np.array([x + w / 2, y + h / 2])
    s = float(h) if h > 0 else 1.0
    q = (P - c) / s
    r = (T - c) / s
    Xd = np.hstack([q, np.ones((n, 1))])
    lam = np.diag([reg * n, reg * n, 0.0])
    W0 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    W = np.linalg.solve(Xd.T @ Xd + lam, Xd.T @ r + lam @ W0)
    M = W[:2].T
    b = W[2]
    sv = np.linalg.svd(M, compute_uv=False)
    if not (0.5 <= sv.min() and sv.max() <= 2.0) or np.linalg.det(M) <= 0:
        log.warning("補正行列が極端なため平行移動だけにします (特異値 %s)", np.round(sv, 3))
        return offset, "offset"
    A = np.hstack([M, (c - M @ c + s * b)[:, None]])
    return A, "affine"
