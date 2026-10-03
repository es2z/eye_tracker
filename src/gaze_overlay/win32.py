"""Win32 APIのラッパー(ctypes)。DPI、多重起動の防止、プロセスの監視、ウィンドウ情報、モニター"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LONG_PTR = ctypes.c_ssize_t

ERROR_ALREADY_EXISTS = 183
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000

HWND_TOPMOST = wintypes.HWND(-1)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

# --- 関数の型定義(64bitでハンドルが切り詰められないように restype を明示する) ---
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetForegroundWindow.argtypes = []
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindow.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetClientRect.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetWindowLongPtrW.restype = LONG_PTR
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowLongPtrW.restype = LONG_PTR
user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, LONG_PTR]
user32.SetWindowPos.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT,
]

kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
]


def set_dpi_awareness() -> str:
    """プロセスをモニターごとのDPI対応にする。座標をすべて物理ピクセルで扱うため、起動直後に呼ぶ"""
    try:
        fn = user32.SetProcessDpiAwarenessContext
        fn.restype = wintypes.BOOL
        fn.argtypes = [wintypes.HANDLE]
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if fn(wintypes.HANDLE(-4)):
            return "per_monitor_v2"
    except AttributeError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per_monitor"
    except Exception:
        log.warning("DPI対応の設定に失敗しました")
        return "unaware"


class SingleInstance:
    """名前付きミューテックスで多重起動を防ぐ"""

    def __init__(self, name: str) -> None:
        self.handle = kernel32.CreateMutexW(None, False, name)
        self.already_running = ctypes.get_last_error() == ERROR_ALREADY_EXISTS

    def close(self) -> None:
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None


def watch_process(pid: int, on_exit: Callable[[], None]) -> bool:
    """指定したプロセス(OBS)の終了を待ち、終了したら on_exit を呼ぶ。異常終了でも検知できる"""
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        log.warning("監視するプロセス(PID %d)を開けませんでした", pid)
        return False

    def run() -> None:
        try:
            if kernel32.WaitForSingleObject(handle, INFINITE) == WAIT_OBJECT_0:
                log.info("監視していたプロセス(PID %d)が終了しました", pid)
                on_exit()
        finally:
            kernel32.CloseHandle(handle)

    threading.Thread(target=run, name="watch-pid", daemon=True).start()
    return True


# --- ウィンドウ情報 ---

def foreground_window() -> int | None:
    return user32.GetForegroundWindow() or None


def set_foreground_window(hwnd: int) -> bool:
    return bool(user32.SetForegroundWindow(hwnd))


def window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def process_image_path(pid: int) -> str | None:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
        return None
    finally:
        kernel32.CloseHandle(handle)


def window_exe(hwnd: int) -> str | None:
    path = process_image_path(window_pid(hwnd))
    return os.path.basename(path) if path else None


def window_title(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def is_window(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def is_minimized(hwnd: int) -> bool:
    return bool(user32.IsIconic(hwnd))


def is_own_window(hwnd: int) -> bool:
    return window_pid(hwnd) == os.getpid()


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int] | None:
    """クライアント領域の (x, y, 幅, 高さ) を仮想デスクトップの物理ピクセルで返す"""
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    w, h = rc.right - rc.left, rc.bottom - rc.top
    if w <= 0 or h <= 0:
        return None
    return pt.x, pt.y, w, h


def make_overlay_window(hwnd: int) -> None:
    """クリックを透過し、アクティブにならない最前面のウィンドウにする(ゲームのフォーカスを奪わないため)"""
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
    user32.SetWindowPos(
        hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED
    )


def keep_topmost(hwnd: int) -> None:
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


# --- モニター ---

@dataclass(frozen=True)
class Monitor:
    name: str
    x: int
    y: int
    width: int
    height: int
    is_primary: bool

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "width": self.width, "height": self.height}


def list_monitors() -> list[Monitor]:
    from screeninfo import get_monitors

    return [
        Monitor(m.name or f"monitor{i}", m.x, m.y, m.width, m.height, bool(m.is_primary))
        for i, m in enumerate(get_monitors())
    ]


def find_monitor(spec: str | int) -> Monitor:
    """"primary"、番号(0始まり)、またはデバイス名でモニターを探す。見つからなければプライマリ"""
    monitors = list_monitors()
    if not monitors:
        raise RuntimeError("モニターが見つかりません")
    primary = next((m for m in monitors if m.is_primary), monitors[0])
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        idx = int(spec)
        if 0 <= idx < len(monitors):
            return monitors[idx]
        log.warning("モニター番号 %s がないためプライマリを使います", spec)
        return primary
    if spec == "primary":
        return primary
    for m in monitors:
        if m.name.lower() == str(spec).lower():
            return m
    log.warning("モニター %s が見つからないためプライマリを使います", spec)
    return primary
