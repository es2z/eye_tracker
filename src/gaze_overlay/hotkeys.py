"""グローバルホットキー(RegisterHotKey)。ゲームを操作中でも効く"""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.RegisterHotKey.restype = wintypes.BOOL
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.UnregisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetMessageW.restype = ctypes.c_int
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.PeekMessageW.restype = wintypes.BOOL
user32.PeekMessageW.argtypes = [
    ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT,
]
user32.PostThreadMessageW.restype = wintypes.BOOL
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
kernel32.GetCurrentThreadId.restype = wintypes.DWORD

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
PM_NOREMOVE = 0x0000

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

_MODS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}
_NAMED_KEYS = {
    "pause": 0x13, "scrolllock": 0x91, "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "space": 0x20, "printscreen": 0x2C,
}


def parse_combo(combo: str) -> tuple[int, int]:
    """"ctrl+alt+f9" のような文字列を (修飾キー, 仮想キーコード) に変換する"""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"空のホットキー: {combo!r}")
    *mods, key = parts
    m = 0
    for name in mods:
        if name not in _MODS:
            raise ValueError(f"不明な修飾キー: {name}")
        m |= _MODS[name]
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif key.startswith("num") and key[3:].isdigit() and len(key) == 4:
        vk = 0x60 + int(key[3:])
    elif len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif key in _NAMED_KEYS:
        vk = _NAMED_KEYS[key]
    else:
        raise ValueError(f"不明なキー: {key}")
    return m, vk


class HotkeyManager:
    def __init__(self, bindings: dict[str, str], callback: Callable[[str], None]) -> None:
        self.bindings = bindings
        self.callback = callback
        # 名前 → {"combo": ..., "ok": bool, "error": str}
        self.results: dict[str, dict] = {}
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(2)

    def _run(self) -> None:
        self._tid = kernel32.GetCurrentThreadId()
        msg = wintypes.MSG()
        # PostThreadMessageを受け取れるよう、先にメッセージキューを作る
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_NOREMOVE)
        ids: dict[int, str] = {}
        for i, (name, combo) in enumerate(self.bindings.items(), start=1):
            if not combo:
                continue
            try:
                mods, vk = parse_combo(combo)
            except ValueError as e:
                self.results[name] = {"combo": combo, "ok": False, "error": str(e)}
                log.warning("ホットキー %s (%s) を解釈できません: %s", name, combo, e)
                continue
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                ids[i] = name
                self.results[name] = {"combo": combo, "ok": True, "error": ""}
            else:
                err = ctypes.get_last_error()
                reason = "他のアプリが使用中" if err == 1409 else f"エラー {err}"
                self.results[name] = {"combo": combo, "ok": False, "error": reason}
                log.warning("ホットキー %s (%s) を登録できません: %s", name, combo, reason)
        log.info("ホットキーを登録しました: %s", ", ".join(f"{n}={self.bindings[n]}" for n in ids.values()))
        self._ready.set()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam in ids:
                    try:
                        self.callback(ids[msg.wParam])
                    except Exception:
                        log.exception("ホットキーの処理でエラー")
        finally:
            for i in ids:
                user32.UnregisterHotKey(None, i)

    def stop(self) -> None:
        if self._tid:
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2)
