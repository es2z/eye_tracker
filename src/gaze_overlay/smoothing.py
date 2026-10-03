"""One Euroフィルタ。じっと見ているときはブレを強く抑え、速く動いたときは遅れを小さくする

参考: Casiez, Roussel, Vogel "1€ Filter" (CHI 2012)
"""

from __future__ import annotations

import math
import threading

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """2次元の座標を平滑化する。速度は大きさ(ノルム)で扱い、縦横で同じ強さにする。
    カメラスレッドが呼び、メインスレッドがリセットや設定変更をするため、ロックで守る"""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self._lock = threading.Lock()
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx = np.zeros(2)
        self._t: float | None = None

    def configure(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        with self._lock:
            self.min_cutoff = min_cutoff
            self.beta = beta
            self.d_cutoff = d_cutoff

    def reset(self) -> None:
        with self._lock:
            self._x = None
            self._dx = np.zeros(2)
            self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        with self._lock:
            if self._x is None or self._t is None:
                self._x = x.copy()
                self._t = t
                return x.copy()
            dt = t - self._t
            if dt <= 0:
                return self._x.copy()
            self._t = t
            dx = (x - self._x) / dt
            a_d = _alpha(self.d_cutoff, dt)
            self._dx = a_d * dx + (1.0 - a_d) * self._dx
            cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
            a = _alpha(cutoff, dt)
            self._x = a * x + (1.0 - a) * self._x
            return self._x.copy()
