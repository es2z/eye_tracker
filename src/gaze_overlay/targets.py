"""視線座標の基準にする矩形(ゲームウィンドウのクライアント領域、またはモニター)を追跡する"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from gaze_overlay import win32
from gaze_overlay.config import TargetConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Target:
    mode: str
    # 仮想デスクトップ上の (x, y, 幅, 高さ) [物理px]
    rect: tuple[int, int, int, int]
    exe: str = ""
    title: str = ""
    hwnd: int = 0

    def to_dict(self) -> dict:
        x, y, w, h = self.rect
        return {
            "mode": self.mode,
            "rect": list(self.rect),
            "exe": self.exe,
            "title": self.title,
            "aspect": w / h if h else 0,
        }


class TargetTracker:
    def __init__(self, cfg: TargetConfig) -> None:
        self.cfg = cfg
        self.mode = cfg.mode
        self._patterns = [p.lower() for p in cfg.patterns]
        self._insets = {k.lower(): v for k, v in cfg.insets.items()}
        self._lock = threading.Lock()
        self._current: Target | None = None
        self._hwnd: int | None = None
        self._pinned: int | None = None
        self._listeners: list[Callable[[Target | None], None]] = []
        self.monitor = win32.find_monitor(cfg.monitor)
        self._monitor_checked = time.monotonic()

    @property
    def current(self) -> Target | None:
        with self._lock:
            return self._current

    def on_change(self, cb: Callable[[Target | None], None]) -> None:
        self._listeners.append(cb)

    def matches(self, exe: str | None) -> bool:
        if not exe:
            return False
        name = exe.lower()
        return any(fnmatch.fnmatchcase(name, p) for p in self._patterns)

    def pin_foreground(self) -> Target | None:
        """今の前面ウィンドウを、exeパターンに関係なく対象として固定する"""
        hwnd = win32.foreground_window()
        if not hwnd or win32.is_own_window(hwnd):
            log.info("前面ウィンドウがないか自分のウィンドウなので、固定しません")
            return self.current
        self._pinned = hwnd
        self.mode = "window"
        log.info("対象を固定しました: %s「%s」", win32.window_exe(hwnd), win32.window_title(hwnd))
        return self.poll()

    def set_mode(self, mode: str) -> None:
        if mode in ("window", "monitor"):
            self.mode = mode
            self.poll()

    def _monitor_target(self) -> Target:
        m = self.monitor
        return Target("monitor", m.rect, title=m.name)

    @staticmethod
    def _valid(hwnd: int | None) -> bool:
        return bool(hwnd) and win32.is_window(hwnd) and not win32.is_minimized(hwnd)

    def _window_target(self, hwnd: int) -> Target | None:
        rect = win32.client_rect_on_screen(hwnd)
        if rect is None:
            return None
        exe = win32.window_exe(hwnd) or ""
        x, y, w, h = rect
        inset = self._insets.get(exe.lower())
        if inset and len(inset) == 4:
            left, top, right, bottom = inset
            x, y, w, h = x + left, y + top, w - left - right, h - top - bottom
            if w <= 0 or h <= 0:
                return None
        return Target("window", (x, y, w, h), exe=exe, title=win32.window_title(hwnd), hwnd=hwnd)

    def poll(self) -> Target | None:
        now = time.monotonic()
        if now - self._monitor_checked > 5.0:
            # 解像度やモニター構成の変更に追従する
            self._monitor_checked = now
            try:
                self.monitor = win32.find_monitor(self.cfg.monitor)
            except Exception:
                log.exception("モニターの再取得に失敗しました")

        target: Target | None = None
        if self.mode == "monitor":
            target = self._monitor_target()
        else:
            hwnd: int | None = None
            if self._pinned and self._valid(self._pinned):
                hwnd = self._pinned
            else:
                self._pinned = None
                fg = win32.foreground_window()
                if fg and fg != self._hwnd and not win32.is_own_window(fg) and self.matches(win32.window_exe(fg)):
                    self._hwnd = fg
                # 前面がブラウザなどに移っても、対象のウィンドウが残っている間は対象のままにする
                if not self._valid(self._hwnd):
                    self._hwnd = None
                hwnd = self._hwnd
            if hwnd:
                target = self._window_target(hwnd)
            if target is None and self.cfg.fallback == "monitor":
                target = self._monitor_target()

        with self._lock:
            changed = target != self._current
            self._current = target
        if changed:
            if target is None:
                log.info("対象: なし")
            else:
                log.info("対象: %s %s「%s」%s", target.mode, target.exe, target.title, target.rect)
            for cb in self._listeners:
                try:
                    cb(target)
                except Exception:
                    log.exception("対象の変更通知でエラー")
        return target

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.poll()
            except Exception:
                log.exception("対象の追跡でエラー")
            stop.wait(self.cfg.poll_ms / 1000)
