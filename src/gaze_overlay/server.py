"""HTTP(オーバーレイとドックのページ)とWebSocket(/ws)を1つのポートで提供する

127.0.0.1 にだけバインドし、WebSocketは同一オリジン(とOriginなしのローカルクライアント)だけ受け付ける。
普段使うブラウザで開いた外部のページから、操作コマンドを送られるのを防ぐため。
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from gaze_overlay.config import WEB_DIR, ServerConfig

log = logging.getLogger(__name__)

PAGES = {"/": "overlay.html", "/overlay": "overlay.html", "/dock": "dock.html"}
STATIC = {"overlay.js", "overlay.css", "dock.js", "dock.css"}


@dataclass
class ClientInfo:
    role: str = "unknown"
    viewport: list[int] | None = None
    # OBSのソースが表示されているか(obsSourceVisibleChanged)。不明なら None
    visible: bool | None = None
    extra: dict = field(default_factory=dict)


class Hub:
    def __init__(
        self,
        cfg: ServerConfig,
        on_command: Callable[[str, Any], None],
        hello_provider: Callable[[str], dict],
        status_provider: Callable[[], dict],
    ) -> None:
        self.cfg = cfg
        self.on_command = on_command
        self.hello_provider = hello_provider
        self.status_provider = status_provider
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: dict[ServerConnection, ClientInfo] = {}
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop: asyncio.Future | None = None
        self._thread: threading.Thread | None = None
        base = [f"http://127.0.0.1:{cfg.port}", f"http://localhost:{cfg.port}"]
        self.origins = [None, *base]

    @property
    def url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    # --- 起動と停止 ---

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, name="server", daemon=True)
        self._thread.start()
        self._ready.wait(5)
        if self._error is not None:
            log.error("サーバーを起動できませんでした: %s", self._error)
            return False
        log.info("サーバーを起動しました: %s/overlay , %s/dock", self.url, self.url)
        return True

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as e:  # ポートが使用中など
            self._error = e
            self._ready.set()

    async def _main(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._stop = self.loop.create_future()
        async with serve(
            self._handler,
            self.cfg.host,
            self.cfg.port,
            process_request=self._process_request,
            origins=self.origins,
            ping_interval=20,
            max_size=2**16,
        ):
            self._ready.set()
            status_task = asyncio.create_task(self._status_loop())
            await self._stop
            status_task.cancel()

    def stop(self) -> None:
        if self.loop is not None and self._stop is not None and not self._stop.done():
            self.loop.call_soon_threadsafe(self._stop.set_result, None)
        if self._thread is not None:
            self._thread.join(timeout=3)

    # --- HTTP ---

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        path = urlsplit(request.path).path
        if path == "/ws":
            return None  # WebSocketのハンドシェイクへ進む
        if path in PAGES:
            name = PAGES[path]
        elif path.startswith("/static/") and path[len("/static/"):] in STATIC:
            name = path[len("/static/"):]
        else:
            return connection.respond(404, "Not Found\n")
        try:
            body = (WEB_DIR / name).read_bytes()  # 編集をすぐ反映できるよう毎回読む
        except FileNotFoundError:
            return connection.respond(404, "Not Found\n")
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        headers = Headers(
            {"Content-Type": ctype, "Content-Length": str(len(body)), "Cache-Control": "no-store"}
        )
        return Response(200, "OK", headers, body)

    # --- WebSocket ---

    async def _handler(self, ws: ServerConnection) -> None:
        info = ClientInfo()
        with self._lock:
            self._clients[ws] = info
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = msg.get("type")
                if kind == "hello":
                    info.role = str(msg.get("role", "unknown"))
                    info.viewport = msg.get("viewport")
                    log.info("接続: %s %s", info.role, info.viewport or "")
                    await ws.send(json.dumps(self.hello_provider(info.role), ensure_ascii=False))
                elif kind == "viewport":
                    info.viewport = msg.get("viewport")
                elif kind == "visibility":
                    info.visible = bool(msg.get("visible"))
                elif kind == "cmd":
                    # 操作はドックとローカルの制御クライアント(--stop)からだけ受け付ける
                    if info.role in ("dock", "control"):
                        self.on_command(str(msg.get("name")), msg.get("value"))
        except ConnectionClosed:
            pass
        finally:
            with self._lock:
                self._clients.pop(ws, None)
            log.info("切断: %s", info.role)

    def _snapshot(self) -> list[tuple[ServerConnection, ClientInfo]]:
        with self._lock:
            return list(self._clients.items())

    def overlays(self) -> list[ClientInfo]:
        return [c for _, c in self._snapshot() if c.role == "overlay"]

    def active_overlay_count(self) -> int:
        """接続中で、OBS上で非表示になっていないオーバーレイの数"""
        return sum(1 for c in self.overlays() if c.visible is not False)

    def _send_role(self, role: str, text: str) -> None:
        conns = [ws for ws, c in self._snapshot() if c.role == role]
        if conns:
            broadcast(conns, text)

    def publish(self, role: str, msg: dict) -> None:
        """他のスレッドから、指定したロールのクライアント全員に送る"""
        if self.loop is None:
            return
        text = json.dumps(msg, ensure_ascii=False)
        try:
            self.loop.call_soon_threadsafe(self._send_role, role, text)
        except RuntimeError:
            pass  # 終了処理中

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            if any(c.role == "dock" for _, c in self._snapshot()):
                try:
                    self._send_role("dock", json.dumps(self.status_provider(), ensure_ascii=False))
                except Exception:
                    log.exception("状態の送信でエラー")
