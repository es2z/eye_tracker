"""設定の読み込み。config.toml(ユーザーが編集)と data/state.json(実行中に変えた値)を扱う"""

from __future__ import annotations

import json
import logging
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# プロジェクトのルート(src/gaze_overlay/config.py から2つ上)。環境変数で上書きできる
ROOT = Path(os.environ.get("GAZE_OVERLAY_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
WEB_DIR = ROOT / "web"
STATE_FILE = DATA_DIR / "state.json"
DEFAULT_CONFIG_FILE = ROOT / "config.toml"


def _default_fallback_resolutions() -> list[list[int]]:
    return [[1920, 1080], [1280, 720], [640, 480]]


@dataclass
class CameraConfig:
    # カメラ名(DirectShowの名前。例 "Camera (NVIDIA Broadcast)")。指定すると index と backend より優先する
    name: str = ""
    index: int = 0
    # "msmf" または "dshow"。開けなかったときはもう一方を試す
    backend: str = "msmf"
    width: int = 640
    height: int = 480
    fps: int = 30
    # 要求した解像度で映像が来ないときに試す解像度(仮想カメラは出力と同じ解像度でないと黒い画像になるため)
    fallback_resolutions: list[list[int]] = field(default_factory=_default_fallback_resolutions)
    # この幅より大きいフレームは縮小してから推論する
    process_width: int = 640
    # 読み込みに失敗し続けたときに開き直す間隔(秒)
    reopen_interval_s: float = 2.0
    # 映像がこの秒数変わらなければ「止まっている」とみなす
    stale_check_s: float = 3.0


@dataclass
class TrackerConfig:
    model: str = "ridge"
    ridge_alpha: float = 1.0
    # 顔ランドマークのモデル。空ならeyetraxのキャッシュ(~/.cache/eyetrax)を使う
    face_model: str = "models/face_landmarker.task"
    # 目の開き具合(EAR)が「開いているときの基準値 × この比率」を下回ったら瞬きとみなす
    blink_threshold_ratio: float = 0.6
    # 瞬き判定が long_blink_ms 以上続いても EAR がこの比率より上なら、目を閉じたのではなく
    # 画面の下を見ている(まぶたが下がっている)とみなして追跡を続ける
    closed_floor_ratio: float = 0.35
    long_blink_ms: int = 400
    # 瞬き直後に位置の更新を止めるフレーム数(まぶたが開ききるまで特徴量が乱れるため)
    post_blink_skip_frames: int = 2
    # 見失った後、このフレーム数だけ連続で顔を検出したら復帰する
    reacquire_frames: int = 3
    # これ以上見失ったら平滑化をリセットする(古い位置から滑ってくるのを防ぐ)
    reset_after_lost_ms: int = 500
    # オーバーレイもドックも接続していないときの推論FPS
    idle_fps: float = 5.0
    # 頭の位置ずれの警告閾値。shiftは両目の間隔を1とした移動量、scaleは距離の変化率
    drift_shift_warn: float = 0.6
    drift_scale_warn: float = 0.12


def _default_presets() -> dict[str, dict[str, float]]:
    # One Euroフィルタのパラメータ(単位はモニターのpx)。Phase 3で実測して調整する。
    # 標準偏差40pxのノイズを入れたシミュレーションでの目安:
    #   stable     静止時のブレ 約9px、400pxの移動で90%に達するまで約300ms
    #   balance    静止時のブレ 約12px、同 約200ms(1000pxの大きな移動は約130ms)
    #   responsive 静止時のブレ 約17px、同 約130ms
    return {
        "stable": {"min_cutoff": 0.3, "beta": 0.0005, "d_cutoff": 1.0},
        "balance": {"min_cutoff": 0.5, "beta": 0.001, "d_cutoff": 1.0},
        "responsive": {"min_cutoff": 1.0, "beta": 0.003, "d_cutoff": 1.0},
    }


@dataclass
class SmoothingConfig:
    preset: str = "balance"
    presets: dict[str, dict[str, float]] = field(default_factory=_default_presets)


@dataclass
class CalibrationConfig:
    # キャリブレーションするモニター。"primary"、番号(0始まり)、または "\\.\DISPLAY3" のようなデバイス名
    monitor: str | int = "primary"
    # 9点キャリブレーションの点を、モニターの端からこの比率だけ内側に置く
    margin: float = 0.10
    pulse_s: float = 1.0
    capture_s: float = 1.0
    # フルキャリブレーションの後に4点で誤差を測る
    validate: bool = True
    validate_capture_s: float = 0.8
    # クイック補正(5点)の点を、対象の端からこの比率だけ内側に置く
    quick_margin: float = 0.15
    quick_settle_s: float = 0.5
    quick_capture_s: float = 0.8
    # 1点オフセット補正
    offset_settle_s: float = 0.6
    offset_capture_s: float = 0.9
    # キャリブレーションのデータがないとき、起動時にフルキャリブレーションを始める
    auto_calibrate_if_missing: bool = True
    file: str = "data/calibration/latest.npz"


def _default_patterns() -> list[str]:
    # OBSのシーンにあるゲームから作った既定のexeパターン(fnmatch、大文字小文字は区別しない)
    return [
        # 東方Project(th06.exe〜th20.exe、th128.exe、th06nc.exe など)。"th*.exe" だと thorium.exe(ブラウザ)にも一致するため数字で始まるものに限る
        "th[0-9]*.exe",
        # 東方の二次創作(th_sohw.exe、th_dlw.exe など)
        "th_*.exe",
        "東方紅魔郷.exe",
        "xenia_canary.exe",
        "Kengeki.exe",
        "Kokishin.exe",
        "THMHJhp.exe",
        "alcostg.exe",
        "LotusLandVReloaded.exe",
        "Illusion.exe",
        "ルナイン3.exe",
        "3rd_IKUSAAAAAAAN.exe",
        "MZZXLC.exe",
        "Proteus.exe",
        "DRAGON QUEST XI S.exe",
        # "game.exe" はOBSのシーン内でも複数のゲーム(Ray'z Arcade Chronology、Touhou BB)が使っており、
        # 汎用的すぎて誤検出しやすいので既定では無効にしている。必要なら config.toml に追加する
        # "game.exe",
    ]


@dataclass
class TargetConfig:
    # "window"(ゲームウィンドウ追従)または "monitor"(モニター全体)
    mode: str = "window"
    monitor: str | int = "primary"
    # 対象のウィンドウがないとき: "hide"(バブルを消す)または "monitor"(モニター全体を対象にする)
    fallback: str = "hide"
    poll_ms: int = 200
    patterns: list[str] = field(default_factory=_default_patterns)
    # exe名ごとのクライアント領域の補正 [左, 上, 右, 下](px、内側が正)
    insets: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


def _default_bindings() -> dict[str, str]:
    return {
        "cycle_preset": "ctrl+alt+f7",
        "pick_foreground": "ctrl+alt+f8",
        "toggle_visible": "ctrl+alt+f9",
        "pause": "ctrl+alt+f10",
        "calibrate_quick": "ctrl+alt+f11",
        "calibrate_full": "ctrl+alt+shift+f11",
        "offset": "ctrl+alt+f12",
    }


@dataclass
class HotkeysConfig:
    enabled: bool = True
    bindings: dict[str, str] = field(default_factory=_default_bindings)


@dataclass
class UiConfig:
    tray: bool = True
    font: str = "Yu Gothic UI"


@dataclass
class LogConfig:
    level: str = "INFO"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def resolve(self, rel: str) -> Path:
        """設定内の相対パスをプロジェクトのルート基準で解決する"""
        p = Path(rel).expanduser()
        return p if p.is_absolute() else ROOT / p


def _merge(obj: Any, data: dict[str, Any], path: str = "") -> None:
    """TOMLの辞書をdataclassに上書きする。未知のキーは警告して無視する"""
    names = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            log.warning("設定の未知のキーを無視します: %s%s", path, key)
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{path}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict) and key in ("presets",):
            # プリセットは既定値に追加・上書きする(全部書かなくてよいように)
            for name, params in value.items():
                current.setdefault(name, {}).update(params)
        else:
            setattr(obj, key, value)


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or DEFAULT_CONFIG_FILE
    if path.exists():
        with path.open("rb") as fh:
            _merge(cfg, tomllib.load(fh))
        log.info("設定を読み込みました: %s", path)
    else:
        log.info("設定ファイルがないため既定値を使います: %s", path)
    return cfg


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        log.exception("state.json を読めませんでした。無視します")
        return {}


def save_state(state: dict[str, Any]) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_FILE)
    except Exception:
        log.exception("state.json を保存できませんでした")
