// 視線バブルの描画(OBSのブラウザソース用)
//
// URLパラメータ(すべて省略可):
//   style      ring(既定、ソフトリング) | bubble(塗りつぶし) | glow(放射グラデーション)
//   size       直径。ブラウザソースの高さに対する比率(既定 0.12)
//   color      色。ff4fa3 のように # なしで書ける(既定 ffffff)
//   opacity    不透明度(既定 0.7)
//   ring       輪の太さ。半径に対する比率(既定 0.16、style=ring のとき)
//   blur       ぼかし。半径に対する比率(既定 0.10)
//   fill       輪の内側の不透明度(既定 0.06、style=ring のとき)
//   edge       領域の外へのはみ出しの許容量。半径を1とした値(既定 1.0)
//   fadeOut    フェードアウトの時間 ms(既定 180)
//   fadeIn     フェードインの時間 ms(既定 120)
//   follow     位置の補間の時定数 ms(既定 40。30Hzのデータを滑らかにつなぐ)
//   trail      1で軌跡を表示(既定 0)
//   trailMs    軌跡の長さ ms(既定 300)
//   trailCount 軌跡の残像の数(既定 8)
//   debug      1で状態を表示(既定 0)
//   ws         WebSocketの接続先(既定 このページと同じホストの /ws)
(() => {
  "use strict";

  const q = new URLSearchParams(location.search);
  const num = (key, def) => {
    const v = parseFloat(q.get(key));
    return Number.isFinite(v) ? v : def;
  };
  const P = {
    style: q.get("style") || "ring",
    size: num("size", 0.12),
    color: q.get("color") || "ffffff",
    opacity: num("opacity", 0.7),
    ring: num("ring", 0.16),
    blur: num("blur", 0.10),
    fill: num("fill", 0.06),
    edge: num("edge", 1.0),
    fadeOut: Math.max(num("fadeOut", 180), 1),
    fadeIn: Math.max(num("fadeIn", 120), 1),
    follow: Math.max(num("follow", 40), 1),
    trail: num("trail", 0) > 0,
    trailMs: num("trailMs", 300),
    trailCount: Math.max(1, Math.round(num("trailCount", 8))),
    debug: num("debug", 0) > 0,
    ws: q.get("ws") || `ws://${location.host}/ws`,
  };

  // 色を [r, g, b] に変換する(#なしの16進数と、CSSの色名の両方を受け付ける)
  function parseColor(text) {
    const css = /^[0-9a-f]{3,8}$/i.test(text) ? `#${text}` : text;
    const probe = document.createElement("div");
    probe.style.color = css;
    document.body.appendChild(probe);
    const m = getComputedStyle(probe).color.match(/\d+(\.\d+)?/g) || ["255", "255", "255"];
    probe.remove();
    return m.slice(0, 3).map(Number);
  }
  const [cr, cg, cb] = parseColor(P.color);
  const rgba = (a) => `rgba(${cr}, ${cg}, ${cb}, ${a})`;

  const stage = document.getElementById("stage");
  const debugEl = document.getElementById("debug");
  debugEl.hidden = !P.debug;

  // バブル本体と軌跡の残像(同じ見た目の要素を使い回す)
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const ghosts = [];
  if (P.trail) {
    for (let i = 0; i < P.trailCount; i++) {
      const g = document.createElement("div");
      g.className = "bubble";
      stage.appendChild(g);
      ghosts.push(g);
    }
  }
  stage.appendChild(bubble);

  let radius = 0;
  function applyStyle(el) {
    const H = window.innerHeight;
    const D = P.size * H;
    radius = D / 2;
    el.style.width = `${D}px`;
    el.style.height = `${D}px`;
    el.style.border = "none";
    el.style.background = "none";
    el.style.filter = "none";
    const blur = P.blur * radius;
    if (P.style === "bubble") {
      el.style.background = rgba(P.opacity * 0.5);
      el.style.filter = `blur(${blur}px)`;
    } else if (P.style === "glow") {
      el.style.background =
        `radial-gradient(circle, ${rgba(P.opacity)} 0%, ${rgba(P.opacity * 0.5)} 35%, ${rgba(0)} 70%)`;
    } else {
      el.style.border = `${P.ring * radius}px solid ${rgba(P.opacity)}`;
      el.style.background = rgba(P.fill);
      el.style.filter = `blur(${blur}px)`;
    }
  }
  function applyAll() {
    applyStyle(bubble);
    ghosts.forEach(applyStyle);
  }
  applyAll();

  // --- 受信した状態 ---
  let connected = false;
  let gaze = null; // {u, v}
  let wanted = false;
  let state = "disconnected";
  let lastMsgAt = 0;
  let target = null;
  let msgCount = 0;
  let msgRate = 0;

  // --- 表示の状態 ---
  let pos = null; // 画面上の位置 {x, y} [px]
  let alpha = 0;
  const history = []; // 軌跡用 {t, x, y}

  let ws = null;
  let backoff = 250;
  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }
  function connect() {
    ws = new WebSocket(P.ws);
    ws.onopen = () => {
      connected = true;
      backoff = 250;
      send({ type: "hello", role: "overlay", viewport: [window.innerWidth, window.innerHeight] });
    };
    ws.onmessage = (ev) => {
      let m;
      try {
        m = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (m.type === "gaze") {
        lastMsgAt = performance.now();
        msgCount++;
        state = m.state;
        wanted = !!m.visible;
        if (m.u !== null && m.v !== null) gaze = { u: m.u, v: m.v };
      } else if (m.type === "hello" || m.type === "target") {
        target = m.target;
      }
    };
    ws.onclose = () => {
      connected = false;
      // Pythonが後から起動した場合や再起動した場合に備えて、つなぎ直し続ける
      setTimeout(connect, backoff);
      backoff = Math.min(2000, backoff * 2);
    };
    ws.onerror = () => {
      try {
        ws.close();
      } catch {
        /* 何もしない */
      }
    };
  }
  connect();

  window.addEventListener("resize", () => {
    applyAll();
    send({ type: "viewport", viewport: [window.innerWidth, window.innerHeight] });
  });
  // OBSでソースの表示・非表示が切り替わったことをPythonに伝える(非表示の間は推論の頻度を下げる)
  window.addEventListener("obsSourceVisibleChanged", (ev) => {
    send({ type: "visibility", visible: !!(ev.detail && ev.detail.visible) });
  });

  setInterval(() => {
    msgRate = msgCount;
    msgCount = 0;
  }, 1000);

  function place(el, x, y, a) {
    el.style.transform = `translate3d(${x - radius}px, ${y - radius}px, 0)`;
    el.style.opacity = a.toFixed(3);
  }

  let last = performance.now();
  function frame() {
    const now = performance.now();
    // 経過時間で動かす(ブラウザソースのFPSに依存しないように)
    const dt = Math.min(100, now - last);
    last = now;
    const W = window.innerWidth;
    const H = window.innerHeight;

    let show = connected && wanted && gaze !== null && now - lastMsgAt < 1000;
    let inside = false;
    if (gaze) {
      const tx = gaze.u * W;
      const ty = gaze.v * H;
      const e = P.edge * radius;
      inside = tx >= -e && tx <= W + e && ty >= -e && ty <= H + e;
      show = show && inside;
      if (pos === null || alpha < 0.02) {
        // 見えていない間は目標へ直接移す(再表示のときに古い位置から滑ってこないように)
        pos = { x: tx, y: ty };
      } else {
        const k = 1 - Math.exp(-dt / P.follow);
        pos.x += (tx - pos.x) * k;
        pos.y += (ty - pos.y) * k;
      }
    }
    alpha = show ? Math.min(1, alpha + dt / P.fadeIn) : Math.max(0, alpha - dt / P.fadeOut);

    if (pos) {
      place(bubble, pos.x, pos.y, alpha);
      if (ghosts.length) {
        history.push({ t: now, x: pos.x, y: pos.y });
        while (history.length && now - history[0].t > P.trailMs + 50) history.shift();
        ghosts.forEach((g, i) => {
          const tt = now - ((i + 1) * P.trailMs) / ghosts.length;
          const h = history.find((p) => p.t >= tt) || history[0];
          place(g, h.x, h.y, alpha * 0.45 * (1 - (i + 1) / (ghosts.length + 1)));
        });
      }
    }

    if (P.debug) {
      const lines = [
        `接続: ${connected ? "OK" : "未接続"}  状態: ${state}  受信: ${msgRate}/s`,
        `u,v: ${gaze ? `${gaze.u.toFixed(3)}, ${gaze.v.toFixed(3)}` : "-"}  領域内: ${inside}  表示: ${alpha.toFixed(2)}`,
        `ソース: ${W}x${H} (${(W / H).toFixed(3)})`,
      ];
      if (target) {
        const ratio = W / H / target.aspect;
        lines.push(`対象: ${target.mode} ${target.exe || target.title} ${target.rect.join(",")} (${target.aspect.toFixed(3)})`);
        if (Math.abs(ratio - 1) > 0.02) lines.push("※ ブラウザソースと対象の縦横比が違います");
      }
      debugEl.textContent = lines.join("\n");
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
