"""キャリブレーションの描画(tkinter)。メインスレッドからだけ使う

- FullscreenSurface: フルキャリブレーション用の黒い全画面。Escで中止できる
- OverlaySurface: 透明でクリックを透過し、前面を奪わない最前面ウィンドウ。
  クイック補正・1点補正のマーカーと、デバッグ用の視線カーソルを描く
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Callable

from gaze_overlay import win32

DOT = "#3ddc84"
RING = "#ffffff"
OUTLINE = "#101010"


class Surface:
    def __init__(self, root: tk.Tk, font_family: str) -> None:
        self.root = root
        self.font_family = font_family
        self.origin = (0, 0)
        self.size = (1, 1)
        self.canvas: tk.Canvas

    def to_local(self, gx: float, gy: float) -> tuple[float, float]:
        return gx - self.origin[0], gy - self.origin[1]

    def clear(self) -> None:
        self.canvas.delete("all")

    def draw_target(self, gx: float, gy: float, phase: str, progress: float, t: float) -> None:
        """点を描く。phase は "settle"(点滅して視線を誘導)または "capture"(収集中、輪が縮む)"""
        x, y = self.to_local(gx, gy)
        c = self.canvas
        if phase == "settle":
            r = 12 + 10 * abs(math.sin(2 * math.pi * t))
            c.create_oval(x - r - 2, y - r - 2, x + r + 2, y + r + 2, fill=OUTLINE, outline="")
            c.create_oval(x - r, y - r, x + r, y + r, fill=DOT, outline="")
        else:
            r = 14
            c.create_oval(x - r - 2, y - r - 2, x + r + 2, y + r + 2, fill=OUTLINE, outline="")
            c.create_oval(x - r, y - r, x + r, y + r, fill=DOT, outline="")
            R = 34
            extent = -359.9 * (1.0 - progress)
            c.create_arc(
                x - R, y - R, x + R, y + R, start=90, extent=extent, style="arc", outline=OUTLINE, width=7
            )
            c.create_arc(
                x - R, y - R, x + R, y + R, start=90, extent=extent, style="arc", outline=RING, width=4
            )
        # 点の中心に小さな黒い点を置き、視線を一点に集めやすくする
        c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=OUTLINE, outline="")

    def draw_message(self, text: str, sub: str = "", color: str = "#ffffff") -> None:
        w, h = self.size
        self.canvas.create_text(
            w / 2, h / 2 - 20, text=text, fill=color, font=(self.font_family, 28, "bold")
        )
        if sub:
            self.canvas.create_text(w / 2, h / 2 + 30, text=sub, fill="#bbbbbb", font=(self.font_family, 16))

    def draw_cursor(self, gx: float, gy: float, raw: tuple[float, float] | None, label: str) -> None:
        """デバッグ用の視線カーソル。輪が平滑化後、小さな点が平滑化前"""
        x, y = self.to_local(gx, gy)
        c = self.canvas
        r = 40
        c.create_oval(x - r, y - r, x + r, y + r, outline=OUTLINE, width=6)
        c.create_oval(x - r, y - r, x + r, y + r, outline="#ff4fa3", width=3)
        if raw is not None:
            rx, ry = self.to_local(*raw)
            c.create_oval(rx - 4, ry - 4, rx + 4, ry + 4, fill="#ffd400", outline=OUTLINE)
        if label:
            c.create_text(
                x + r + 8, y - r, text=label, anchor="nw", fill="#ffffff", font=(self.font_family, 11)
            )


class FullscreenSurface(Surface):
    def __init__(
        self, root: tk.Tk, font_family: str, rect: tuple[int, int, int, int], on_escape: Callable[[], None]
    ) -> None:
        super().__init__(root, font_family)
        x, y, w, h = rect
        self.origin = (x, y)
        self.size = (w, h)
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.configure(bg="black")
        self.top.geometry(f"{w}x{h}+{x}+{y}")
        self.top.attributes("-topmost", True)
        self.canvas = tk.Canvas(self.top, width=w, height=h, bg="black", highlightthickness=0, cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.top.bind("<Escape>", lambda _e: on_escape())
        self.top.update()
        self.top.focus_force()

    def close(self) -> None:
        self.top.destroy()


class OverlaySurface(Surface):
    # 透過色(この色で塗った部分は透明になる)
    KEY = "#010203"

    def __init__(self, root: tk.Tk, font_family: str) -> None:
        super().__init__(root, font_family)
        prev = win32.foreground_window()
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.configure(bg=self.KEY)
        self.top.attributes("-transparentcolor", self.KEY)
        self.top.attributes("-topmost", True)
        self.top.geometry("1x1+0+0")
        self.canvas = tk.Canvas(self.top, width=1, height=1, bg=self.KEY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.top.update()
        self.hwnd = int(self.top.wm_frame(), 16)
        win32.make_overlay_window(self.hwnd)
        # 表示した瞬間にフォーカスを取った場合に備えて、元の前面ウィンドウへ戻す
        if prev and not win32.is_own_window(prev):
            win32.set_foreground_window(prev)
        self.active = False

    def cover(self, rect: tuple[int, int, int, int]) -> None:
        """指定した範囲を覆う。一度表示したウィンドウを動かすだけなので、アクティブにならない"""
        x, y, w, h = rect
        if self.active and self.origin == (x, y) and self.size == (w, h):
            return
        self.origin = (x, y)
        self.size = (w, h)
        self.top.geometry(f"{w}x{h}+{x}+{y}")
        self.canvas.configure(width=w, height=h)
        win32.keep_topmost(self.hwnd)
        self.active = True

    def shrink(self) -> None:
        """使わないときは1x1に縮める。ゲーム画面に重なったままだと、フルスクリーン最適化が効かなくなることがあるため"""
        self.clear()
        x, y = self.origin
        self.top.geometry(f"1x1+{x}+{y}")
        self.size = (1, 1)
        self.active = False
