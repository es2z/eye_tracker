"""タスクトレイのアイコンとメニュー(pystray)"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

PRESET_LABELS = {"stable": "安定", "balance": "バランス", "responsive": "追従"}


def _make_icon(color: tuple[int, int, int, int]):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), outline=color, width=9)
    d.ellipse((26, 26, 38, 38), fill=color)
    return img


class Tray:
    def __init__(self, command: Callable[[str, Any], None], state: Callable[[], dict]) -> None:
        import pystray
        from pystray import Menu
        from pystray import MenuItem as Item

        self._pystray = pystray
        self.command = command
        self.state = state
        self._icons = {
            "ok": _make_icon((255, 255, 255, 255)),
            "warn": _make_icon((255, 200, 0, 255)),
            "off": _make_icon((140, 140, 140, 255)),
        }
        self._icon_kind = "off"

        def cmd(name: str, value: Any = None):
            return lambda _icon, _item: self.command(name, value)

        def preset_item(name: str):
            return Item(
                PRESET_LABELS.get(name, name),
                cmd("set_preset", name),
                checked=lambda _i: self.state().get("preset") == name,
                radio=True,
            )

        presets = self.state().get("presets", list(PRESET_LABELS))
        menu = Menu(
            Item("表示する", cmd("toggle_visible"), checked=lambda _i: self.state().get("shown", True)),
            Item("一時停止", cmd("pause"), checked=lambda _i: self.state().get("paused", False)),
            Menu.SEPARATOR,
            Item("1点オフセット補正", cmd("offset")),
            Item("クイック補正(5点)", cmd("calibrate_quick")),
            Item("フルキャリブレーション(9点)", cmd("calibrate_full")),
            Menu.SEPARATOR,
            Item("平滑化", Menu(*[preset_item(p) for p in presets])),
            Item("前面ウィンドウを対象にする(3秒後)", cmd("pick_foreground", 3)),
            Item(
                "モニター全体を対象にする",
                lambda _icon, _item: self.command(
                    "set_mode", "window" if self.state().get("mode") == "monitor" else "monitor"
                ),
                checked=lambda _i: self.state().get("mode") == "monitor",
            ),
            Menu.SEPARATOR,
            Item("ドックのURLをコピー", cmd("copy_dock_url")),
            Item("ログフォルダを開く", cmd("open_logs")),
            Menu.SEPARATOR,
            Item("終了", cmd("shutdown")),
        )
        self.icon = pystray.Icon("gaze_overlay", self._icons["off"], "視線オーバーレイ", menu)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.icon.run, name="tray", daemon=True)
        self._thread.start()

    def refresh(self, title: str, kind: str) -> None:
        """ツールチップ、アイコンの色、メニューのチェック状態を更新する(メインスレッドから1秒ごと)"""
        try:
            self.icon.title = title[:120]
            if kind != self._icon_kind and kind in self._icons:
                self._icon_kind = kind
                self.icon.icon = self._icons[kind]
            self.icon.update_menu()
        except Exception:
            log.debug("トレイの更新に失敗", exc_info=True)

    def notify(self, message: str, title: str = "視線オーバーレイ") -> None:
        try:
            self.icon.notify(message, title)
        except Exception:
            log.debug("トレイの通知に失敗", exc_info=True)

    def stop(self) -> None:
        try:
            self.icon.stop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2)
