// OBSのカスタムブラウザドック用の操作パネル
(() => {
  "use strict";

  const STATE_LABELS = {
    tracking: ["追跡中", "ok"],
    blink: ["瞬き", "ok"],
    lost: ["顔なし", "warn"],
    paused: ["一時停止", "warn"],
    calibrating: ["キャリブレーション中", "warn"],
    uncalibrated: ["未キャリブレーション", "bad"],
    no_camera: ["カメラなし", "bad"],
    starting: ["起動中", "warn"],
    error: ["エラー", "bad"],
  };
  const PRESET_LABELS = { stable: "安定", balance: "バランス", responsive: "追従" };
  const HOTKEY_LABELS = {
    cycle_preset: "平滑化切替",
    pick_foreground: "前面を対象",
    toggle_visible: "表示切替",
    pause: "一時停止",
    calibrate_quick: "クイック補正",
    calibrate_full: "フル",
    offset: "1点補正",
  };

  const $ = (id) => document.getElementById(id);
  let ws = null;
  let backoff = 250;
  let last = null;

  function send(name, value = null) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "cmd", name, value }));
  }

  document.querySelectorAll("button[data-cmd]").forEach((b) => {
    b.addEventListener("click", () => {
      const name = b.dataset.cmd;
      if (name === "set_mode") send(name, last && last.mode === "monitor" ? "window" : "monitor");
      else send(name, b.dataset.value != null ? Number(b.dataset.value) : null);
    });
  });

  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => {
      backoff = 250;
      ws.send(JSON.stringify({ type: "hello", role: "dock" }));
    };
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "status") render(m);
    };
    ws.onclose = () => {
      last = null;
      $("badge").textContent = "未接続";
      $("badge").className = "badge bad";
      $("perf").textContent = "Pythonが起動していません";
      setTimeout(connect, backoff);
      backoff = Math.min(2000, backoff * 2);
    };
    ws.onerror = () => ws.close();
  }

  function fmtAge(sec) {
    if (sec < 120) return `${Math.round(sec)}秒前`;
    if (sec < 7200) return `${Math.round(sec / 60)}分前`;
    if (sec < 172800) return `${Math.round(sec / 3600)}時間前`;
    return `${Math.round(sec / 86400)}日前`;
  }

  function renderPresets(s) {
    const box = $("presets");
    if (box.dataset.names !== s.presets.join(",")) {
      box.dataset.names = s.presets.join(",");
      box.innerHTML = "";
      s.presets.forEach((name) => {
        const b = document.createElement("button");
        b.textContent = PRESET_LABELS[name] || name;
        b.dataset.preset = name;
        b.addEventListener("click", () => send("set_preset", name));
        box.appendChild(b);
      });
    }
    box.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.preset === s.preset));
  }

  function render(s) {
    last = s;
    const [label, cls] = STATE_LABELS[s.state] || [s.state, "warn"];
    $("badge").textContent = label;
    $("badge").className = `badge ${cls}`;
    $("perf").textContent =
      `${s.fps}fps 処理${s.proc_ms}ms CPU ${s.cpu_pct}%(1コア比) / ${s.cpu_total_pct}%(全体)` + (s.idle ? " 省電力" : "");

    $("btn-visible").classList.toggle("on", s.shown);
    $("btn-visible").textContent = s.shown ? "表示中" : "非表示";
    $("btn-pause").classList.toggle("on", s.paused);
    $("btn-mode").textContent = s.mode === "monitor" ? "対象: モニター" : "対象: ウィンドウ";
    renderPresets(s);

    const t = s.target;
    $("target").textContent = t
      ? `${t.mode === "window" ? t.exe : t.title}  ${t.rect[2]}x${t.rect[3]} @${t.rect[0]},${t.rect[1]}`
      : "なし(対象のゲームを前面にしてください)";

    const d = s.head_drift;
    if (!s.head_available) {
      $("drift").textContent = "取得できません";
    } else if (d) {
      const pct = Math.min(100, (d.score / 1.5) * 100);
      $("drift-bar").style.width = `${pct}%`;
      $("drift-bar").style.background = d.warn ? "var(--warn)" : "var(--ok)";
      $("drift").textContent = d.warn ? "ずれています(補正推奨)" : "OK";
    } else {
      $("drift-bar").style.width = "0";
      $("drift").textContent = "-";
    }

    const c = s.calib;
    if (!c.exists) {
      $("calib").textContent = "なし(フルキャリブレーションを実行してください)";
    } else {
      const err = c.error_px != null ? `誤差 ${Math.round(c.error_px)}px (${c.error_pct.toFixed(1)}%)` : "誤差 -";
      const corr = c.correction && c.correction !== "none" ? ` 補正:${c.correction}` : "";
      $("calib").textContent = `${fmtAge(c.age_s)} ${err}${corr}`;
    }

    $("eye").textContent =
      s.ear != null ? `EAR ${s.ear} / 基準 ${s.ear_baseline ?? "-"}${s.looking_down ? " (下を見ている)" : ""}` : "-";

    const cam = s.camera;
    if (!cam || !cam.open) {
      $("camera").textContent = cam && cam.error ? `開いていません: ${cam.error}` : "開いていません";
    } else {
      $("camera").textContent =
        `${cam.name || cam.backend.toUpperCase()} ${cam.width}x${cam.height}` +
        (cam.process_width !== cam.width ? ` → ${cam.process_width}x${cam.process_height}` : "") +
        (cam.stale ? " (映像が止まっています)" : "");
    }

    const hk = Object.entries(s.hotkeys || {});
    $("hotkeys").innerHTML = "";
    hk.forEach(([name, r]) => {
      const span = document.createElement("div");
      span.textContent = `${r.ok ? "✓" : "✗"} ${r.combo} ${HOTKEY_LABELS[name] || name}${r.ok ? "" : ` (${r.error})`}`;
      span.className = r.ok ? "" : "muted";
      $("hotkeys").appendChild(span);
    });
    if (!hk.length) $("hotkeys").textContent = "無効";

    const warnings = [];
    if (t) {
      (s.overlays || []).forEach((o) => {
        if (!o.viewport) return;
        const [w, h] = o.viewport;
        if (Math.abs(w / h / t.aspect - 1) > 0.02) {
          warnings.push(
            `ブラウザソース(${w}x${h})と対象(${t.rect[2]}x${t.rect[3]})の縦横比が違います。` +
              "ソースの幅と高さをゲームキャプチャの元のサイズに合わせてください",
          );
        }
      });
    }
    if (!(s.overlays || []).length) warnings.push("オーバーレイ(ブラウザソース)が接続されていません");
    if (cam && cam.open && cam.stale) {
      warnings.push("カメラの映像が止まっています。NVIDIA Broadcastが起動しているか確認してください");
    }
    if (d && d.warn) warnings.push("頭の位置がキャリブレーション時からずれています。1点補正かクイック補正を行ってください");
    $("warnings").innerHTML = "";
    warnings.forEach((w) => {
      const div = document.createElement("div");
      div.textContent = w;
      $("warnings").appendChild(div);
    });
  }

  connect();
})();
