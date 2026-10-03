"""カメラの入力

既定は物理カメラを MSMF(ハードウェア変換オフ)で開く。
VMagicMirror などとカメラを共有する場合は、NVIDIA Broadcast の仮想カメラ「Camera (NVIDIA Broadcast)」を
DirectShow で名前指定して開く(Windowsのマルチアプリカメラは DirectShow のアプリと共有できないため)。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import gaze_overlay  # noqa: F401  環境変数の設定を cv2 より先に済ませるため
import cv2
import numpy as np

from gaze_overlay.config import CameraConfig

log = logging.getLogger(__name__)

_BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}

# 開いた直後に映像が来ているか確かめる時間(秒)。仮想カメラは解像度が合わないと黒い画像を1秒に1枚しか出さない
_VERIFY_S = 1.0
_VERIFY_MIN_FPS = 5.0
_DARK_LEVEL = 8.0


def list_dshow_devices() -> list[str]:
    """DirectShowのカメラ名を、OpenCVの番号と同じ順番で返す"""
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        log.exception("DirectShowのカメラ一覧を取得できませんでした")
        return []


@dataclass
class CameraInfo:
    backend: str = ""
    name: str = ""
    index: int = -1
    width: int = 0
    height: int = 0
    fps: float = 0.0
    process_width: int = 0
    process_height: int = 0
    open_ms: float = 0.0
    # 映像が止まっている(仮想カメラの元のアプリが起動していないときの静止画など)
    stale: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Camera:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap: cv2.VideoCapture | None = None
        self.info = CameraInfo()
        self._fail_count = 0
        self._next_reopen = 0.0
        self._frame_no = 0
        self._last_thumb: np.ndarray | None = None
        self._static_since: float | None = None

    @property
    def is_open(self) -> bool:
        return self.cap is not None

    # ------------------------------------------------------------ 開く

    def _targets(self) -> list[tuple[str, int, str]]:
        """試す (バックエンド, 番号, 名前) の順番"""
        if self.cfg.name:
            # 名前で指定したときは DirectShow の番号を引く(MSMFとは番号の順番が違うため、MSMFでは試さない)
            names = list_dshow_devices()
            matches = [i for i, n in enumerate(names) if n.lower() == self.cfg.name.lower()]
            if not matches:
                msg = f"カメラ「{self.cfg.name}」が見つかりません(候補: {', '.join(names) or 'なし'})"
                log.error(msg)
                self.info.error = msg
                return []
            return [("dshow", matches[0], self.cfg.name)]
        order = [self.cfg.backend] + [b for b in _BACKENDS if b != self.cfg.backend]
        return [(b, self.cfg.index, "") for b in order if b in _BACKENDS]

    def _resolutions(self) -> list[tuple[int, int]]:
        first = (self.cfg.width, self.cfg.height)
        rest = [tuple(r) for r in self.cfg.fallback_resolutions if tuple(r) != first]
        return [first, *rest]

    @staticmethod
    def _verify(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None, float, float]:
        """映像が実際に来ているか確かめる。(合格, 最後のフレーム, FPS, 平均輝度) を返す"""
        n = 0
        frame = None
        level = 0.0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < _VERIFY_S:
            ok, f = cap.read()
            if ok and f is not None:
                n += 1
                frame = f
                level = float(f[::16, ::16].mean())
        fps = n / (time.perf_counter() - t0)
        return (frame is not None and fps >= _VERIFY_MIN_FPS and level > _DARK_LEVEL), frame, fps, level

    def open(self) -> bool:
        """設定のカメラを開く。映像が来ない解像度なら、候補の解像度を順に試す"""
        self.release()
        self.info.error = ""
        for backend, index, name in self._targets():
            for w, h in self._resolutions():
                t0 = time.perf_counter()
                cap = cv2.VideoCapture(index, _BACKENDS[backend])
                if not cap.isOpened():
                    cap.release()
                    log.warning("カメラ %s を %s で開けませんでした", name or index, backend)
                    break  # 解像度を変えても開けないので次のバックエンドへ
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                ok, frame, fps, level = self._verify(cap)
                if not ok:
                    cap.release()
                    log.warning(
                        "カメラ %s (%s %dx%d) から映像が来ません: %.1ffps 平均輝度 %.0f",
                        name or index, backend, w, h, fps, level,
                    )
                    continue
                assert frame is not None
                fh, fw = frame.shape[:2]
                pw, ph = self._process_size(fw, fh)
                self.info = CameraInfo(
                    backend=backend,
                    name=name,
                    index=index,
                    width=fw,
                    height=fh,
                    fps=round(fps, 1),
                    process_width=pw,
                    process_height=ph,
                    open_ms=(time.perf_counter() - t0) * 1000,
                )
                self.cap = cap
                self._fail_count = 0
                self._last_thumb = None
                self._static_since = None
                log.info(
                    "カメラを開きました: %s %s %dx%d (要求 %dx%d@%d) 推論 %dx%d、%.0fms",
                    backend, name or f"#{index}", fw, fh, w, h, self.cfg.fps, pw, ph, self.info.open_ms,
                )
                return True
        if not self.info.error:
            self.info.error = "カメラから映像が来ません(解像度が合わないか、映像の元のアプリが起動していない可能性)"
        self._next_reopen = time.monotonic() + self.cfg.reopen_interval_s
        return False

    def _process_size(self, w: int, h: int) -> tuple[int, int]:
        if w <= self.cfg.process_width:
            return w, h
        scale = self.cfg.process_width / w
        return self.cfg.process_width, round(h * scale)

    # ------------------------------------------------------------ 読む

    def read(self) -> np.ndarray | None:
        """1フレーム読み、必要なら縮小して返す。失敗が続いたら一定間隔で開き直す"""
        if self.cap is None:
            if time.monotonic() >= self._next_reopen:
                self.open()
            if self.cap is None:
                time.sleep(0.05)
                return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._fail_count += 1
            if self._fail_count >= 30:
                log.warning("カメラの読み込みに失敗し続けたため開き直します")
                self.release()
                self._next_reopen = time.monotonic() + self.cfg.reopen_interval_s
            else:
                time.sleep(0.01)
            return None
        self._fail_count = 0
        self._check_stale(frame)
        h, w = frame.shape[:2]
        if w > self.cfg.process_width:
            frame = cv2.resize(frame, self._process_size(w, h), interpolation=cv2.INTER_AREA)
        return frame

    def _check_stale(self, frame: np.ndarray) -> None:
        """映像が止まっていないか、15フレームごとに縮小画像を比べて調べる。
        本物のカメラは必ずノイズで少し変わるので、まったく同じなら静止画とみなす"""
        self._frame_no += 1
        if self._frame_no % 15:
            return
        thumb = frame[::32, ::32].astype(np.int16)
        now = time.monotonic()
        if self._last_thumb is not None and self._last_thumb.shape == thumb.shape:
            if float(np.abs(thumb - self._last_thumb).mean()) < 0.2:
                if self._static_since is None:
                    self._static_since = now
            else:
                self._static_since = None
        self._last_thumb = thumb
        stale = self._static_since is not None and now - self._static_since >= self.cfg.stale_check_s
        if stale != self.info.stale:
            self.info.stale = stale
            if stale:
                log.warning("カメラの映像が止まっています(仮想カメラの元のアプリが起動していない可能性)")
            else:
                log.info("カメラの映像が戻りました")

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
